from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
class FixedString(ValueField):
    structvalues = 1

    def __init__(self, name, size):
        ValueField.__init__(self, name)
        self.structcode = '%ds' % size