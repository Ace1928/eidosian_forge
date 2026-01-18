from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
def writeTTCHeader(file, numFonts):
    self = SimpleNamespace()
    self.TTCTag = 'ttcf'
    self.Version = 65536
    self.numFonts = numFonts
    file.seek(0)
    file.write(sstruct.pack(ttcHeaderFormat, self))
    offset = file.tell()
    file.write(struct.pack('>%dL' % self.numFonts, *[0] * self.numFonts))
    return offset