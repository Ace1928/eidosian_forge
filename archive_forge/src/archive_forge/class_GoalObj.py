import ctypes
class GoalObj(ctypes.c_void_p):

    def __init__(self, goal):
        self._as_parameter_ = goal

    def from_param(obj):
        return obj