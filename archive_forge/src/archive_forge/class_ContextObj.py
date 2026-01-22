import ctypes
class ContextObj(ctypes.c_void_p):

    def __init__(self, context):
        self._as_parameter_ = context

    def from_param(obj):
        return obj