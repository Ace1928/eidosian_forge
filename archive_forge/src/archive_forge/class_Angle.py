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
class Angle(F2Dot14):
    bias = 0.0
    factor = 1.0 / (1 << 14) * 180

    @classmethod
    def fromInt(cls, value):
        return (super().fromInt(value) + cls.bias) * 180

    @classmethod
    def toInt(cls, value):
        return super().toInt(value / 180 - cls.bias)

    @classmethod
    def fromString(cls, value):
        return otRound(float(value) / cls.factor) * cls.factor

    @classmethod
    def toString(cls, value):
        return nearestMultipleShortestRepr(value, cls.factor)