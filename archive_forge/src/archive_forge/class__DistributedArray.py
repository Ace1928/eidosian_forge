import cupy
import numpy
class _DistributedArray(cupy.ndarray):

    def __new__(cls, shape, dtype, chunks, axis, tile_size, devices):
        mem = _MultiDeviceDummyMemory(0)
        memptr = _MultiDeviceDummyPointer(mem, 0)
        obj = super().__new__(cls, shape, dtype, memptr=memptr)
        obj._chunks = chunks
        obj._tile_size = tile_size
        obj._mem = mem
        obj._axis = axis
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._chunks = getattr(obj, 'chunks', None)
        self._tile_size = getattr(obj, 'tile_size', None)
        self._axis = getattr(obj, 'axis', None)
        self._mem = getattr(obj, 'mem', None)

    def _get_chunk(self, i):
        return self._chunks[i]

    def _prepare_args(self, dist_args, regular_args, device):
        args = []
        c_shape = None
        for i, arg in dist_args:
            chunk = arg._get_chunk(device)
            args.append((i, chunk))
            if c_shape is None:
                c_shape = chunk.shape
            if chunk.shape != c_shape:
                raise RuntimeError('Operating distributed arrays of different chunk sizes together is not supported')
        if len(regular_args) > 0:
            raise RuntimeError('Mix `cupy.ndarray` with distributed arrays is not currentlysupported')
        return args

    def _get_execution_devices(self, dist_args):
        devices = set()
        for _, arg in dist_args:
            for dev in arg._chunks:
                devices.add(dev)
        return devices

    def _execute_kernel(self, kernel, args, kwargs):
        distributed_arrays = []
        regular_arrays = []
        for i, arg in enumerate(args):
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((i, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((i, arg))
        for k, arg in kwargs.items():
            if isinstance(arg, _DistributedArray):
                distributed_arrays.append((k, arg))
            elif isinstance(arg, cupy.ndarray):
                regular_arrays.append((k, arg))
        args = list(args)
        devices = self._get_execution_devices(distributed_arrays)
        dev_outs = {}
        dtype = None
        for dev in devices:
            array_args = self._prepare_args(distributed_arrays, regular_arrays, dev)
            for i, arg in array_args:
                if isinstance(i, int):
                    args[i] = arg
                else:
                    kwargs[i] = arg
            with cupy.cuda.Device(dev):
                out = kernel(*args, **kwargs)
                dtype = out.dtype
                dev_outs[dev] = out
        for out in dev_outs.values():
            if not isinstance(out, cupy.ndarray):
                raise RuntimeError('kernels returning other than single array not supported')
        return _DistributedArray(self.shape, dtype, dev_outs, self._axis, self._tile_size, devices)

    def __cupy_override_elementwise_kernel__(self, kernel, *args, **kwargs):
        outs = self._execute_kernel(kernel, args, kwargs)
        return outs

    def asnumpy(self):
        chunks = [cupy.asnumpy(c) for c in self._chunks.values()]
        return numpy.concatenate(chunks, axis=self._axis)