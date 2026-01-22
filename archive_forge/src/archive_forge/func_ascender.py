from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from fontTools.misc.fixedTools import (
from . import DefaultTable
import math
@ascender.setter
def ascender(self, value):
    self.ascent = value