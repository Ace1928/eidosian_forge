import ctypes
class AstVectorObj(ctypes.c_void_p):

    def __init__(self, vector):
        self._as_parameter_ = vector

    def from_param(obj):
        return obj