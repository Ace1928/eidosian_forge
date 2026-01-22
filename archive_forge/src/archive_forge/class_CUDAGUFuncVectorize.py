from numba import cuda
from numpy import array as np_array
from numba.np.ufunc import deviceufunc
from numba.np.ufunc.deviceufunc import (UFuncMechanism, GeneralizedUFunc,
class CUDAGUFuncVectorize(deviceufunc.DeviceGUFuncVectorize):

    def build_ufunc(self):
        engine = deviceufunc.GUFuncEngine(self.inputsig, self.outputsig)
        return CUDAGeneralizedUFunc(kernelmap=self.kernelmap, engine=engine, pyfunc=self.pyfunc)

    def _compile_kernel(self, fnobj, sig):
        return cuda.jit(sig)(fnobj)

    @property
    def _kernel_template(self):
        return _gufunc_stager_source

    def _get_globals(self, sig):
        corefn = cuda.jit(sig, device=True)(self.pyfunc)
        glbls = self.py_func.__globals__.copy()
        glbls.update({'__cuda__': cuda, '__core__': corefn})
        return glbls