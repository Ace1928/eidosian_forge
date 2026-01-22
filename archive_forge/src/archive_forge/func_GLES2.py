import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def GLES2(self):
    try:
        return ctypesloader.loadLibrary(ctypes.cdll, 'GLESv2', mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None