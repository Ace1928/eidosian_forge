import sys
import ctypes
from pyglet.util import debug_print
def adjustor_cb_func(p, *args):
    assert _debug_com(f'COMObject method {method_name} called through interface {interface_name}, adjusting pointer by {offset}')
    return original_method(p + offset, *args)