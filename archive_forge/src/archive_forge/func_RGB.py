import ctypes
def RGB(red, green, blue):
    return red + (green << 8) + (blue << 16)