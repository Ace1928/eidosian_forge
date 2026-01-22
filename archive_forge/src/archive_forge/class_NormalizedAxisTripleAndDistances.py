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
@dataclasses.dataclass(frozen=True, order=True, repr=False)
class NormalizedAxisTripleAndDistances(AxisTriple):
    """A triple of (min, default, max) normalized axis values,
    with distances between min and default, and default and max,
    in the *pre-normalized* space."""
    minimum: float
    default: float
    maximum: float
    distanceNegative: Optional[float] = 1
    distancePositive: Optional[float] = 1

    def __post_init__(self):
        if self.default is None:
            object.__setattr__(self, 'default', max(self.minimum, min(self.maximum, 0)))
        if not -1.0 <= self.minimum <= self.default <= self.maximum <= 1.0:
            raise ValueError(f'Normalized axis values not in -1..+1 range; got minimum={self.minimum:g}, default={self.default:g}, maximum={self.maximum:g})')

    def reverse_negate(self):
        v = self
        return self.__class__(-v[2], -v[1], -v[0], v[4], v[3])

    def renormalizeValue(self, v, extrapolate=True):
        """Renormalizes a normalized value v to the range of this axis,
        considering the pre-normalized distances as well as the new
        axis limits."""
        lower, default, upper, distanceNegative, distancePositive = self
        assert lower <= default <= upper
        if not extrapolate:
            v = max(lower, min(upper, v))
        if v == default:
            return 0
        if default < 0:
            return -self.reverse_negate().renormalizeValue(-v, extrapolate=extrapolate)
        if v > default:
            return (v - default) / (upper - default)
        if lower >= 0:
            return (v - default) / (default - lower)
        totalDistance = distanceNegative * -lower + distancePositive * default
        if v >= 0:
            vDistance = (default - v) * distancePositive
        else:
            vDistance = -v * distanceNegative + distancePositive * default
        return -vDistance / totalDistance