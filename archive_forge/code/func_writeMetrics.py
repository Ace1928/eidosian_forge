from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def writeMetrics(self, writer, ttFont):
    self.metrics.toXML(writer, ttFont)