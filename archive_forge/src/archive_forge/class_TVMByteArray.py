import ctypes
class TVMByteArray(ctypes.Structure):
    """Temp data structure for byte array."""
    _fields_ = [('data', ctypes.POINTER(ctypes.c_byte)), ('size', ctypes.c_size_t)]