from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
def writeMasterChecksum(self, directory):
    checksumadjustment = self._calcMasterChecksum(directory)
    self.file.seek(self.tables['head'].offset + 8)
    self.file.write(struct.pack('>L', checksumadjustment))