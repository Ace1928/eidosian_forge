from numba.np.numpy_support import from_dtype
def _gpu_reduce_factory(fn, nbtype):
    from numba import cuda
    reduce_op = cuda.jit(device=True)(fn)
    inner_sm_size = _WARPSIZE + 1
    max_blocksize = _NUMWARPS * _WARPSIZE

    @cuda.jit(device=True)
    def inner_warp_reduction(sm_partials, init):
        """
        Compute reduction within a single warp
        """
        tid = cuda.threadIdx.x
        warpid = tid // _WARPSIZE
        laneid = tid % _WARPSIZE
        sm_this = sm_partials[warpid, :]
        sm_this[laneid] = init
        cuda.syncwarp()
        width = _WARPSIZE // 2
        while width:
            if laneid < width:
                old = sm_this[laneid]
                sm_this[laneid] = reduce_op(old, sm_this[laneid + width])
            cuda.syncwarp()
            width //= 2

    @cuda.jit(device=True)
    def device_reduce_full_block(arr, partials, sm_partials):
        """
        Partially reduce `arr` into `partials` using `sm_partials` as working
        space.  The algorithm goes like:

            array chunks of 128:  |   0 | 128 | 256 | 384 | 512 |
                        block-0:  |   x |     |     |   x |     |
                        block-1:  |     |   x |     |     |   x |
                        block-2:  |     |     |   x |     |     |

        The array is divided into chunks of 128 (size of a threadblock).
        The threadblocks consumes the chunks in roundrobin scheduling.
        First, a threadblock loads a chunk into temp memory.  Then, all
        subsequent chunks are combined into the temp memory.

        Once all chunks are processed.  Inner-block reduction is performed
        on the temp memory.  So that, there will just be one scalar result
        per block.  The result from each block is stored to `partials` at
        the dedicated slot.
        """
        tid = cuda.threadIdx.x
        blkid = cuda.blockIdx.x
        blksz = cuda.blockDim.x
        gridsz = cuda.gridDim.x
        start = tid + blksz * blkid
        stop = arr.size
        step = blksz * gridsz
        tmp = arr[start]
        for i in range(start + step, stop, step):
            tmp = reduce_op(tmp, arr[i])
        cuda.syncthreads()
        inner_warp_reduction(sm_partials, tmp)
        cuda.syncthreads()
        if tid < 2:
            sm_partials[tid, 0] = reduce_op(sm_partials[tid, 0], sm_partials[tid + 2, 0])
            cuda.syncwarp()
        if tid == 0:
            partials[blkid] = reduce_op(sm_partials[0, 0], sm_partials[1, 0])

    @cuda.jit(device=True)
    def device_reduce_partial_block(arr, partials, sm_partials):
        """
        This computes reduction on `arr`.
        This device function must be used by 1 threadblock only.
        The blocksize must match `arr.size` and must not be greater than 128.
        """
        tid = cuda.threadIdx.x
        blkid = cuda.blockIdx.x
        blksz = cuda.blockDim.x
        warpid = tid // _WARPSIZE
        laneid = tid % _WARPSIZE
        size = arr.size
        tid = cuda.threadIdx.x
        value = arr[tid]
        sm_partials[warpid, laneid] = value
        cuda.syncthreads()
        if (warpid + 1) * _WARPSIZE < size:
            inner_warp_reduction(sm_partials, value)
        elif laneid == 0:
            sm_this = sm_partials[warpid, :]
            base = warpid * _WARPSIZE
            for i in range(1, size - base):
                sm_this[0] = reduce_op(sm_this[0], sm_this[i])
        cuda.syncthreads()
        if tid == 0:
            num_active_warps = (blksz + _WARPSIZE - 1) // _WARPSIZE
            result = sm_partials[0, 0]
            for i in range(1, num_active_warps):
                result = reduce_op(result, sm_partials[i, 0])
            partials[blkid] = result

    def gpu_reduce_block_strided(arr, partials, init, use_init):
        """
        Perform reductions on *arr* and writing out partial reduction result
        into *partials*.  The length of *partials* is determined by the
        number of threadblocks. The initial value is set with *init*.

        Launch config:

        Blocksize must be multiple of warpsize and it is limited to 4 warps.
        """
        tid = cuda.threadIdx.x
        sm_partials = cuda.shared.array((_NUMWARPS, inner_sm_size), dtype=nbtype)
        if cuda.blockDim.x == max_blocksize:
            device_reduce_full_block(arr, partials, sm_partials)
        else:
            device_reduce_partial_block(arr, partials, sm_partials)
        if use_init and tid == 0 and (cuda.blockIdx.x == 0):
            partials[0] = reduce_op(partials[0], init)
    return cuda.jit(gpu_reduce_block_strided)