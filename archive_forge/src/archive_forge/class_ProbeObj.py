import ctypes
class ProbeObj(ctypes.c_void_p):

    def __init__(self, probe):
        self._as_parameter_ = probe

    def from_param(obj):
        return obj