import ctypes
class StatsObj(ctypes.c_void_p):

    def __init__(self, statistics):
        self._as_parameter_ = statistics

    def from_param(obj):
        return obj