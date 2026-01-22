import os
import sys
import threading
from . import process
from . import reduction
def RawValue(self, typecode_or_type, *args):
    """Returns a shared object"""
    from .sharedctypes import RawValue
    return RawValue(typecode_or_type, *args)