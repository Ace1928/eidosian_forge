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
def _setHeadTransformFlag(self):
    """Set bit 11 of 'head' table flags to indicate that the font has undergone
        a lossless modifying transform. Re-compile head table data."""
    self._decompileTable('head')
    self.ttFont['head'].flags |= 1 << 11
    self._compileTable('head')