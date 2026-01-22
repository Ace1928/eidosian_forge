import ctypes, ctypes.util
from OpenGL.platform import baseplatform, ctypesloader
@baseplatform.lazy_property
def GetCurrentContext(self):
    return self.CGL.CGLGetCurrentContext