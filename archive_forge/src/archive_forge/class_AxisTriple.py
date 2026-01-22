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
class AxisTriple(Sequence):
    """A triple of (min, default, max) axis values.

    Any of the values can be None, in which case the limitRangeAndPopulateDefaults()
    method can be used to fill in the missing values based on the fvar axis values.
    """
    minimum: Optional[float]
    default: Optional[float]
    maximum: Optional[float]

    def __post_init__(self):
        if self.default is None and self.minimum == self.maximum:
            object.__setattr__(self, 'default', self.minimum)
        if self.minimum is not None and self.default is not None and (self.minimum > self.default) or (self.default is not None and self.maximum is not None and (self.default > self.maximum)) or (self.minimum is not None and self.maximum is not None and (self.minimum > self.maximum)):
            raise ValueError(f'{type(self).__name__} minimum ({self.minimum}), default ({self.default}), maximum ({self.maximum}) must be in sorted order')

    def __getitem__(self, i):
        fields = dataclasses.fields(self)
        return getattr(self, fields[i].name)

    def __len__(self):
        return len(dataclasses.fields(self))

    def _replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def __repr__(self):
        return f'({', '.join((format(v, 'g') if v is not None else 'None' for v in self))})'

    @classmethod
    def expand(cls, v: Union['AxisTriple', float, Tuple[float, float], Tuple[float, float, float]]) -> 'AxisTriple':
        """Convert a single value or a tuple into an AxisTriple.

        If the input is a single value, it is interpreted as a pin at that value.
        If the input is a tuple, it is interpreted as (min, max) or (min, default, max).
        """
        if isinstance(v, cls):
            return v
        if isinstance(v, (int, float)):
            return cls(v, v, v)
        try:
            n = len(v)
        except TypeError as e:
            raise ValueError(f'expected float, 2- or 3-tuple of floats; got {type(v)}: {v!r}') from e
        default = None
        if n == 2:
            minimum, maximum = v
        elif n >= 3:
            return cls(*v)
        else:
            raise ValueError(f'expected sequence of 2 or 3; got {n}: {v!r}')
        return cls(minimum, default, maximum)

    def limitRangeAndPopulateDefaults(self, fvarTriple) -> 'AxisTriple':
        """Return a new AxisTriple with the default value filled in.

        Set default to fvar axis default if the latter is within the min/max range,
        otherwise set default to the min or max value, whichever is closer to the
        fvar axis default.
        If the default value is already set, return self.
        """
        minimum = self.minimum
        if minimum is None:
            minimum = fvarTriple[0]
        default = self.default
        if default is None:
            default = fvarTriple[1]
        maximum = self.maximum
        if maximum is None:
            maximum = fvarTriple[2]
        minimum = max(minimum, fvarTriple[0])
        maximum = max(maximum, fvarTriple[0])
        minimum = min(minimum, fvarTriple[2])
        maximum = min(maximum, fvarTriple[2])
        default = max(minimum, min(maximum, default))
        return AxisTriple(minimum, default, maximum)