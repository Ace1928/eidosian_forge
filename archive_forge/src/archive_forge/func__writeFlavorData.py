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
def _writeFlavorData(self):
    """Write metadata and/or private data using appropiate padding."""
    compressedMetaData = self.compressedMetaData
    privData = self.flavorData.privData
    if compressedMetaData and privData:
        compressedMetaData = pad(compressedMetaData, size=4)
    if compressedMetaData:
        self.file.seek(self.metaOffset)
        assert self.file.tell() == self.metaOffset
        self.file.write(compressedMetaData)
    if privData:
        self.file.seek(self.privOffset)
        assert self.file.tell() == self.privOffset
        self.file.write(privData)