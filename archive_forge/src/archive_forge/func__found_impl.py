import sys
import ctypes
from pyglet.util import debug_print
def _found_impl(interface_name, method_name, method_func, self_distance):
    """If a method was found in class, create a callback extracting self from the struct
    pointer.
    """

    def self_extracting_cb_func(p, *args):
        assert _debug_com(f'COMObject method {method_name} called through interface {interface_name}')
        self = ctypes.cast(p + self_distance, ctypes.POINTER(ctypes.py_object)).contents.value
        result = method_func(self, *args)
        return S_OK if result is None else result
    return self_extracting_cb_func