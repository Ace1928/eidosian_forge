from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class GetAttrData(object):

    def __getattr__(self, attr):
        try:
            if self._data:
                return self._data[attr]
            else:
                raise AttributeError(attr)
        except KeyError:
            raise AttributeError(attr)