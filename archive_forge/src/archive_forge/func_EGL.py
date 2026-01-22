import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def EGL(self):
    import os
    if os.path.exists('/proc/cpuinfo'):
        info = open('/proc/cpuinfo').read()
        if 'BCM2708' in info or 'BCM2709' in info:
            assert self.GLES2
    try:
        return ctypesloader.loadLibrary(ctypes.cdll, 'EGL', mode=ctypes.RTLD_GLOBAL)
    except OSError as err:
        raise ImportError('Unable to load EGL library', *err.args)