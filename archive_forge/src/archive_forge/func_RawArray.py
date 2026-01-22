import os
import sys
import threading
from . import process
from . import reduction
def RawArray(self, typecode_or_type, size_or_initializer):
    """Returns a shared array"""
    from .sharedctypes import RawArray
    return RawArray(typecode_or_type, size_or_initializer)