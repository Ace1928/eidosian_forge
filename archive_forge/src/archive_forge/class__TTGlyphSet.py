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
class _TTGlyphSet(Mapping):
    """Generic dict-like GlyphSet class that pulls metrics from hmtx and
    glyph shape from TrueType or CFF.
    """

    def __init__(self, font, location, glyphsMapping, *, recalcBounds=True):
        self.recalcBounds = recalcBounds
        self.font = font
        self.defaultLocationNormalized = {axis.axisTag: 0 for axis in self.font['fvar'].axes} if 'fvar' in self.font else {}
        self.location = location if location is not None else {}
        self.rawLocation = {}
        self.originalLocation = location if location is not None else {}
        self.depth = 0
        self.locationStack = []
        self.rawLocationStack = []
        self.glyphsMapping = glyphsMapping
        self.hMetrics = font['hmtx'].metrics
        self.vMetrics = getattr(font.get('vmtx'), 'metrics', None)
        self.hvarTable = None
        if location:
            from fontTools.varLib.varStore import VarStoreInstancer
            self.hvarTable = getattr(font.get('HVAR'), 'table', None)
            if self.hvarTable is not None:
                self.hvarInstancer = VarStoreInstancer(self.hvarTable.VarStore, font['fvar'].axes, location)

    @contextmanager
    def pushLocation(self, location, reset: bool):
        self.locationStack.append(self.location)
        self.rawLocationStack.append(self.rawLocation)
        if reset:
            self.location = self.originalLocation.copy()
            self.rawLocation = self.defaultLocationNormalized.copy()
        else:
            self.location = self.location.copy()
            self.rawLocation = {}
        self.location.update(location)
        self.rawLocation.update(location)
        try:
            yield None
        finally:
            self.location = self.locationStack.pop()
            self.rawLocation = self.rawLocationStack.pop()

    @contextmanager
    def pushDepth(self):
        try:
            depth = self.depth
            self.depth += 1
            yield depth
        finally:
            self.depth -= 1

    def __contains__(self, glyphName):
        return glyphName in self.glyphsMapping

    def __iter__(self):
        return iter(self.glyphsMapping.keys())

    def __len__(self):
        return len(self.glyphsMapping)

    @deprecateFunction("use 'glyphName in glyphSet' instead", category=DeprecationWarning)
    def has_key(self, glyphName):
        return glyphName in self.glyphsMapping