import ctypes
class SolverObj(ctypes.c_void_p):

    def __init__(self, solver):
        self._as_parameter_ = solver

    def from_param(obj):
        return obj