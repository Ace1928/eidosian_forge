import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
class vectorize(object):
    __doc__ = np.vectorize.__doc__

    def __init__(self, *args, **kwargs):
        self.__dict__['_is_numpy_pyfunc'] = False
        self.__dict__['_cupy_support'] = False
        if isinstance(args[0], _RecursiveAttr):
            self.__dict__['_is_numpy_pyfunc'] = True
            if args[0]._cupy_object:
                self.__dict__['_cupy_support'] = True
            args = (args[0]._numpy_object,) + args[1:]
        notification._dispatch_notification(np.vectorize)
        self.__dict__['vec_obj'] = np.vectorize(*args, **kwargs)
        self.__dict__['__doc__'] = self.__dict__['vec_obj'].__doc__

    def __getattr__(self, attr):
        return getattr(self.__dict__['vec_obj'], attr)

    def __setattr__(self, name, value):
        return setattr(self.vec_obj, name, value)

    def __call__(self, *args, **kwargs):
        if self._is_numpy_pyfunc:
            notification._dispatch_notification(self.vec_obj.pyfunc, self._cupy_support)
        return _call_numpy(self.vec_obj, args, kwargs)