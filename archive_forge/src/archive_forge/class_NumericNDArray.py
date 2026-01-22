from pyomo.common.dependencies import numpy as np, numpy_available
class NumericNDArray(np.ndarray if numpy_available else object):
    """An ndarray subclass that stores Pyomo numeric expressions"""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            args = [np.asarray(i) for i in inputs]
            kwargs['dtype'] = object
        ans = getattr(ufunc, method)(*args, **kwargs)
        if isinstance(ans, np.ndarray):
            if ans.size == 1:
                return ans[0]
            return ans.view(NumericNDArray)
        else:
            return ans