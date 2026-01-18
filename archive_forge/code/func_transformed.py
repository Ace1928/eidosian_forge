from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
@transformed.setter
def transformed(self, booleanValue):
    if self.tag in {'glyf', 'loca'}:
        self.transformVersion = 3 if not booleanValue else 0
    else:
        self.transformVersion = int(booleanValue)