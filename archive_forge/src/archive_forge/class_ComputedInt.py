from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class ComputedInt(IntValue):

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if value is not None:
            xmlWriter.comment('%s=%s' % (name, value))
            xmlWriter.newline()