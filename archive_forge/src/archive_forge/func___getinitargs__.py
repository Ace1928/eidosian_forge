import time as _time
import math as _math
import sys
from operator import index as _index
def __getinitargs__(self):
    """pickle support"""
    if self._name is None:
        return (self._offset,)
    return (self._offset, self._name)