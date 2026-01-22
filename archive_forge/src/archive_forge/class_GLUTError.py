import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
class GLUTError(Error):
    """GLUT error implementation class"""