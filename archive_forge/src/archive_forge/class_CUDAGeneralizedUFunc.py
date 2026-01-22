from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GeneralizedUFunc,
class CUDAGeneralizedUFunc(GeneralizedUFunc):

    def __init__(self, kernelmap, engine, pyfunc):
        self.__name__ = pyfunc.__name__
        super().__init__(kernelmap, engine)

    @property
    def _call_steps(self):
        return _CUDAGUFuncCallSteps

    def _broadcast_scalar_input(self, ary, shape):
        return cuda.cudadrv.devicearray.DeviceNDArray(shape=shape, strides=(0,), dtype=ary.dtype, gpu_data=ary.gpu_data)

    def _broadcast_add_axis(self, ary, newshape):
        newax = len(newshape) - len(ary.shape)
        newstrides = (0,) * newax + ary.strides
        return cuda.cudadrv.devicearray.DeviceNDArray(shape=newshape, strides=newstrides, dtype=ary.dtype, gpu_data=ary.gpu_data)