from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from types import SimpleNamespace
from fontTools.misc.fixedTools import otRound
from fontTools.misc.loggingTools import deprecateFunction
from fontTools.misc.transform import Transform
from fontTools.pens.transformPen import TransformPen, TransformPointPen
from fontTools.pens.recordingPen import (
class _TTGlyphSetCFF(_TTGlyphSet):

    def __init__(self, font, location):
        tableTag = 'CFF2' if 'CFF2' in font else 'CFF '
        self.charStrings = list(font[tableTag].cff.values())[0].CharStrings
        super().__init__(font, location, self.charStrings)
        self.blender = None
        if location:
            from fontTools.varLib.varStore import VarStoreInstancer
            varStore = getattr(self.charStrings, 'varStore', None)
            if varStore is not None:
                instancer = VarStoreInstancer(varStore.otVarStore, font['fvar'].axes, location)
                self.blender = instancer.interpolateFromDeltas

    def __getitem__(self, glyphName):
        return _TTGlyphCFF(self, glyphName)