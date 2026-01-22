import ctypes
class ModelObj(ctypes.c_void_p):

    def __init__(self, model):
        self._as_parameter_ = model

    def from_param(obj):
        return obj