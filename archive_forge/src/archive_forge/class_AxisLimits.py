from fontTools.misc.fixedTools import (
from fontTools.varLib.models import normalizeValue, piecewiseLinearMap
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.TupleVariation import TupleVariation
from fontTools.ttLib.tables import _g_l_y_f
from fontTools import varLib
from fontTools import subset  # noqa: F401
from fontTools.varLib import builder
from fontTools.varLib.mvar import MVAR_ENTRIES
from fontTools.varLib.merger import MutatorMerger
from fontTools.varLib.instancer import names
from .featureVars import instantiateFeatureVariations
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.varLib.instancer import solver
import collections
import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import logging
import os
import re
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
import warnings
class AxisLimits(_BaseAxisLimits):
    """Maps axis tags (str) to AxisTriple values."""

    def __init__(self, *args, **kwargs):
        self._data = data = {}
        for k, v in dict(*args, **kwargs).items():
            if v is None:
                data[k] = v
            else:
                try:
                    triple = AxisTriple.expand(v)
                except ValueError as e:
                    raise ValueError(f'Invalid axis limits for {k!r}: {v!r}') from e
                data[k] = triple

    def limitAxesAndPopulateDefaults(self, varfont) -> 'AxisLimits':
        """Return a new AxisLimits with defaults filled in from fvar table.

        If all axis limits already have defaults, return self.
        """
        fvar = varfont['fvar']
        fvarTriples = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in fvar.axes}
        newLimits = {}
        for axisTag, triple in self.items():
            fvarTriple = fvarTriples[axisTag]
            default = fvarTriple[1]
            if triple is None:
                newLimits[axisTag] = AxisTriple(default, default, default)
            else:
                newLimits[axisTag] = triple.limitRangeAndPopulateDefaults(fvarTriple)
        return type(self)(newLimits)

    def normalize(self, varfont, usingAvar=True) -> 'NormalizedAxisLimits':
        """Return a new NormalizedAxisLimits with normalized -1..0..+1 values.

        If usingAvar is True, the avar table is used to warp the default normalization.
        """
        fvar = varfont['fvar']
        badLimits = set(self.keys()).difference((a.axisTag for a in fvar.axes))
        if badLimits:
            raise ValueError('Cannot limit: {} not present in fvar'.format(badLimits))
        axes = {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in fvar.axes if a.axisTag in self}
        avarSegments = {}
        if usingAvar and 'avar' in varfont:
            avarSegments = varfont['avar'].segments
        normalizedLimits = {}
        for axis_tag, triple in axes.items():
            distanceNegative = triple[1] - triple[0]
            distancePositive = triple[2] - triple[1]
            if self[axis_tag] is None:
                normalizedLimits[axis_tag] = NormalizedAxisTripleAndDistances(0, 0, 0, distanceNegative, distancePositive)
                continue
            minV, defaultV, maxV = self[axis_tag]
            if defaultV is None:
                defaultV = triple[1]
            avarMapping = avarSegments.get(axis_tag, None)
            normalizedLimits[axis_tag] = NormalizedAxisTripleAndDistances(*(normalize(v, triple, avarMapping) for v in (minV, defaultV, maxV)), distanceNegative, distancePositive)
        return NormalizedAxisLimits(normalizedLimits)