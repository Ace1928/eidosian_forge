import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
class CloudpickledObjectWrapper:

    def __init__(self, obj, keep_wrapper=False):
        self._obj = obj
        self._keep_wrapper = keep_wrapper

    def __reduce__(self):
        _pickled_object = dumps(self._obj)
        if not self._keep_wrapper:
            return (loads, (_pickled_object,))
        return (_reconstruct_wrapper, (_pickled_object, self._keep_wrapper))

    def __getattr__(self, attr):
        if attr not in ['_obj', '_keep_wrapper']:
            return getattr(self._obj, attr)
        return getattr(self, attr)