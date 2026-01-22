import ctypes
class FuncDecl(ctypes.c_void_p):

    def __init__(self, decl):
        self._as_parameter_ = decl

    def from_param(obj):
        return obj