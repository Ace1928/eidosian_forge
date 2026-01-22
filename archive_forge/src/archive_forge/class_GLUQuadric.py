from OpenGL.raw import GLU as _simple
from OpenGL.platform import createBaseFunction, PLATFORM
import ctypes
class GLUQuadric(_simple.GLUquadric):
    """Implementation class for GLUQuadric classes in PyOpenGL"""
    FUNCTION_TYPE = PLATFORM.functionTypeFor(PLATFORM.GLU)
    CALLBACK_TYPES = {_simple.GLU_ERROR: FUNCTION_TYPE(None, _simple.GLenum)}

    def addCallback(self, which, function):
        """Register a callback for the quadric object
        
        At the moment only GLU_ERROR is supported by OpenGL, but
        we allow for the possibility of more callbacks in the future...
        """
        callbackType = self.CALLBACK_TYPES.get(which)
        if not callbackType:
            raise ValueError("Don't have a registered callback type for %r" % (which,))
        if not isinstance(function, callbackType):
            cCallback = callbackType(function)
        else:
            cCallback = function
        PLATFORM.GLU.gluQuadricCallback(self, which, cCallback)
        if getattr(self, 'callbacks', None) is None:
            self.callbacks = {}
        self.callbacks[which] = cCallback
        return cCallback