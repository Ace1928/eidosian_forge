from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
class _MemcpySync(_BuiltinFunc):

    def __call__(self, group, dst, dst_idx, src, src_idx, size, *, aligned_size=None):
        """Calls ``cg::memcpy_sync()``.

        Args:
            group: a valid cooperative group
            dst: the destination array that can be viewed as a 1D
                C-contiguous array
            dst_idx: the start index of the destination array element
            src: the source array that can be viewed as a 1D C-contiguous
                array
            src_idx: the start index of the source array element
            size (int): the number of bytes to be copied from
                ``src[src_index]`` to ``dst[dst_idx]``
            aligned_size (int): Use ``cuda::aligned_size_t<N>`` to guarantee
                the compiler that ``src``/``dst`` are at least N-bytes aligned.
                The behavior is undefined if the guarantee is not held.

        .. seealso:: `cg::memcpy_sync`_

        .. _cg::memcpy_sync:
            https://docs.nvidia.com/cuda/archive/11.6.0/cuda-c-programming-guide/index.html#collectives-cg-memcpy-async
        """
        super().__call__()

    def call(self, env, group, dst, dst_idx, src, src_idx, size, *, aligned_size=None):
        if _runtime.runtimeGetVersion() < 11010:
            raise RuntimeError('not supported in CUDA < 11.1')
        _check_include(env, 'cg')
        _check_include(env, 'cg_memcpy_async')
        dst = _Data.init(dst, env)
        src = _Data.init(src, env)
        for arr in (dst, src):
            if not isinstance(arr.ctype, (_cuda_types.CArray, _cuda_types.Ptr)):
                raise TypeError('dst/src must be of array type.')
        dst = _compile._indexing(dst, dst_idx, env)
        src = _compile._indexing(src, src_idx, env)
        size = _compile._astype_scalar(size, _cuda_types.uint32, 'same_kind', env)
        size = _Data.init(size, env)
        size_code = f'{size.code}'
        if aligned_size:
            if not isinstance(aligned_size, _Constant):
                raise ValueError('aligned_size must be a compile-time constant')
            _check_include(env, 'cuda_barrier')
            size_code = f'cuda::aligned_size_t<{aligned_size.obj}>({size_code})'
        return _Data(f'cg::memcpy_async({group.code}, &({dst.code}), &({src.code}), {size_code})', _cuda_types.void)