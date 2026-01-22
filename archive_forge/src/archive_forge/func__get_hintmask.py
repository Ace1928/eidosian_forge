from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
def _get_hintmask(self, index):
    if not self.hintMaskBytes:
        args = self.countHints()
        if args:
            self._hint_op('vstemhm', args)
        self.hintMaskBytes = (self.hintCount + 7) // 8
    hintMaskBytes, index = self.callingStack[-1].getBytes(index, self.hintMaskBytes)
    return (index, hintMaskBytes)