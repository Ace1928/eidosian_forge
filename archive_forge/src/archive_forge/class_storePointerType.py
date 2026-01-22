import OpenGL
import ctypes
from OpenGL import _configflags
from OpenGL import contextdata, error, converters
from OpenGL.arrays import arraydatatype
from OpenGL._bytes import bytes,unicode
import logging
from OpenGL import acceleratesupport
class storePointerType(object):
    """Store named pointer value in context indexed by constant
    
    pointerName -- named pointer argument 
    constant -- constant used to index in the context storage
    
    Note: OpenGL.STORE_POINTERS can be set with ERROR_ON_COPY
    to ignore this storage operation.
    
    Stores the pyArgs (i.e. result of pyConverters) for the named
    pointer argument...
    """

    def __init__(self, pointerName, constant):
        self.pointerName = pointerName
        self.constant = constant

    def finalise(self, wrapper):
        self.pointerIndex = wrapper.pyArgIndex(self.pointerName)

    def __call__(self, result, baseOperation, pyArgs, cArgs):
        contextdata.setValue(self.constant, pyArgs[self.pointerIndex])