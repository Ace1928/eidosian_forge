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
def _getVersion(self):
    """Return the WOFF2 font's (majorVersion, minorVersion) tuple."""
    data = self.flavorData
    if data.majorVersion is not None and data.minorVersion is not None:
        return (data.majorVersion, data.minorVersion)
    elif 'head' in self.tables:
        return struct.unpack('>HH', self.tables['head'].data[4:8])
    else:
        return (0, 0)