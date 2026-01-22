import ctypes
class SimplifierObj(ctypes.c_void_p):

    def __init__(self, simplifier):
        self._as_parameter_ = simplifier

    def from_param(obj):
        return obj