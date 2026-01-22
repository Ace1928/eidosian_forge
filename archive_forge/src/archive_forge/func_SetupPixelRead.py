from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
from OpenGL import arrays
from OpenGL import error
from OpenGL import _configflags
import ctypes
def SetupPixelRead(format, dims, type):
    """Setup transfer mode for a read into a numpy array return the array
    
    Calls setupDefaultTransferMode, sets rankPacking and then 
    returns a createTargetArray for the parameters.
    """
    setupDefaultTransferMode()
    rankPacking(len(dims) + 1)
    return createTargetArray(format, dims, type)