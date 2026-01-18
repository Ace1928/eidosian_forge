from . import _ccallback_c
import ctypes
@property
def user_data(self):
    return tuple.__getitem__(self, 2)