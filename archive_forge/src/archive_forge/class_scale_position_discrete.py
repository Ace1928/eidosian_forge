from __future__ import annotations
import typing
from itertools import chain
import numpy as np
import pandas as pd
from .._utils import array_kind, match
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineError
from ..iapi import range_view
from ._expand import expand_range
from .range import RangeContinuous
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_position_discrete(scale_discrete):
    """
    Base class for discrete position scales

    Parameters
    ----------
    {superclass_parameters}
    limits : array_like, default=None
        Limits of the scale. For discrete scale, these are
        the categories (unique values) of the variable.
        For scales that deal with categoricals, these may
        be a subset or superset of the categories.
    """
    guide = None
    range_c: RangeContinuous

    def __init__(self, *args, **kwargs):
        self.range_c = RangeContinuous()
        scale_discrete.__init__(self, *args, **kwargs)

    def reset(self):
        self.range_c.reset()

    def is_empty(self) -> bool:
        return super().is_empty() and self.range_c.is_empty()

    def train(self, x, drop=False):
        if array_kind.continuous(x):
            self.range_c.train(x)
        else:
            self.range.train(x, drop=self.drop)

    def map(self, x, limits=None):
        if limits is None:
            limits = self.limits
        if array_kind.discrete(x):
            seq = np.arange(1, len(limits) + 1)
            idx = np.asarray(match(x, limits, nomatch=len(x)))
            if not len(idx):
                return []
            try:
                seq = seq[idx]
            except IndexError:
                seq = np.hstack((seq.astype(float), np.nan))
                idx = np.clip(idx, 0, len(seq) - 1)
                seq = seq[idx]
            return list(seq)
        return list(x)

    @property
    def limits(self):
        if self.is_empty():
            return (0, 1)
        elif self._limits is not None and (not callable(self._limits)):
            return self._limits
        elif self._limits is None:
            return self.range.range
        elif callable(self._limits):
            limits = self._limits(self.range.range)
            if iter(limits) is limits:
                limits = list(limits)
            return limits
        else:
            raise PlotnineError('Lost, do not know what the limits are.')

    @limits.setter
    def limits(self, value):
        if isinstance(value, tuple):
            value = list(value)
        self._limits = value

    def dimension(self, expand=(0, 0, 0, 0), limits=None):
        """
        Get the phyical size of the scale

        Unlike limits, this always returns a numeric vector of length 2
        """
        from mizani.bounds import expand_range_distinct
        if limits is None:
            limits = self.limits
        if self.is_empty():
            return (0, 1)
        if self.range.is_empty():
            return expand_range_distinct(self.range_c.range, expand)
        elif self.range_c.is_empty():
            return expand_range_distinct((1, len(self.limits)), expand)
        else:
            a = np.hstack([self.range_c.range, expand_range_distinct((1, len(self.range.range)), expand)])
            return (a.min(), a.max())

    def expand_limits(self, limits: Sequence[str], expand: TupleFloat2 | TupleFloat4, coord_limits: TupleFloat2, trans: trans) -> range_view:
        if self.is_empty():
            climits = (0, 1)
        else:
            climits = (1, len(limits))
            self.range_c.range
        if coord_limits is not None:
            c0, c1 = coord_limits
            climits = (climits[0] if c0 is None else c0, climits[1] if c1 is None else c1)
        rv_d = expand_range(climits, expand, trans)
        if self.range_c.is_empty():
            return rv_d
        no_expand = self.default_expansion(0, 0)
        rv_c = expand_range(self.range_c.range, no_expand, trans)
        rv = range_view(range=(min(chain(rv_d.range, rv_c.range)), max(chain(rv_d.range, rv_c.range))), range_coord=(min(chain(rv_d.range_coord, rv_c.range_coord)), max(chain(rv_d.range_coord, rv_c.range_coord))))
        rv.range = (min(rv.range), max(rv.range))
        rv.range_coord = (min(rv.range_coord), max(rv.range_coord))
        return rv