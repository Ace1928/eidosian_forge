import ctypes
class ParserContextObj(ctypes.c_void_p):

    def __init__(self, pc):
        self._as_parameter_ = pc

    def from_param(obj):
        return obj