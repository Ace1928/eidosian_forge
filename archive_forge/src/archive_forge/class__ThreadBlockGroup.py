from cupy.cuda import runtime as _runtime
from cupyx.jit import _compile
from cupyx.jit import _cuda_types
from cupyx.jit._internal_types import BuiltinFunc as _BuiltinFunc
from cupyx.jit._internal_types import Constant as _Constant
from cupyx.jit._internal_types import Data as _Data
from cupyx.jit._internal_types import wraps_class_method as _wraps_class_method
class _ThreadBlockGroup(_ThreadGroup):
    """A handle to the current thread block group. Must be
    created via :func:`this_thread_block`.

    .. seealso:: `CUDA Thread Block Group API`_

    .. _CUDA Thread Block Group API:
        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-group-cg
    """

    def __init__(self):
        self.child_type = 'cg::thread_block'

    @_wraps_class_method
    def sync(self, env, instance):
        """
        sync()

        Synchronize the threads named in the group.
        """
        return super()._sync(env, instance)

    @_wraps_class_method
    def thread_rank(self, env, instance):
        """
        thread_rank()

        Rank of the calling thread within ``[0, num_threads)``.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.thread_rank()', _cuda_types.uint32)

    @_wraps_class_method
    def group_index(self, env, instance):
        """
        group_index()

        3-Dimensional index of the block within the launched grid.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.group_index()', _cuda_types.dim3)

    @_wraps_class_method
    def thread_index(self, env, instance):
        """
        thread_index()

        3-Dimensional index of the thread within the launched block.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.thread_index()', _cuda_types.dim3)

    @_wraps_class_method
    def dim_threads(self, env, instance):
        """
        dim_threads()

        Dimensions of the launched block in units of threads.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError('dim_threads() is supported on CUDA 11.6+')
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.dim_threads()', _cuda_types.dim3)

    @_wraps_class_method
    def num_threads(self, env, instance):
        """
        num_threads()

        Total number of threads in the group.
        """
        if _runtime.runtimeGetVersion() < 11060:
            raise RuntimeError('num_threads() is supported on CUDA 11.6+')
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.num_threads()', _cuda_types.uint32)

    @_wraps_class_method
    def size(self, env, instance):
        """
        size()

        Total number of threads in the group.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.size()', _cuda_types.uint32)

    @_wraps_class_method
    def group_dim(self, env, instance):
        """
        group_dim()

        Dimensions of the launched block in units of threads.
        """
        _check_include(env, 'cg')
        return _Data(f'{instance.code}.group_dim()', _cuda_types.dim3)