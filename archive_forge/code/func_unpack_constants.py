import os, sys
from OpenGL.plugins import PlatformPlugin
from OpenGL import _configflags
def unpack_constants(constants, namespace):
    """Create constants and add to the namespace"""
    from OpenGL.constant import Constant
    for line in constants.splitlines():
        if line and line.split():
            name, value = line.split()
            namespace[name] = Constant(name, int(value, 16))