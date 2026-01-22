import ctypes
@property
def as_voidp(self):
    return ctypes.c_voidp(self.address)