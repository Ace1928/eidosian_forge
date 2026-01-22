from __future__ import annotations
import copy
from enum import Enum
from packaging.version import Version
import numpy as np
from datashader.datashape import dshape, isnumeric, Record, Option
from datashader.datashape import coretypes as ct
from toolz import concat, unique
import xarray as xr
from datashader.antialias import AntialiasCombination, AntialiasStage2
from datashader.utils import isminus1, isnull
from numba import cuda as nb_cuda
from .utils import (
class summary(Expr):
    """A collection of named reductions.

    Computes all aggregates simultaneously, output is stored as a
    ``xarray.Dataset``.

    Examples
    --------
    A reduction for computing the mean of column "a", and the sum of column "b"
    for each bin, all in a single pass.

    >>> import datashader as ds
    >>> red = ds.summary(mean_a=ds.mean('a'), sum_b=ds.sum('b'))

    Notes
    -----
    A single pass of the source dataset using antialiased lines can either be
    performed using a single-stage aggregation (e.g. ``self_intersect=True``)
    or two stages (``self_intersect=False``). If a ``summary`` contains a
    ``count`` or ``sum`` reduction with ``self_intersect=False``, or any of
    ``first``, ``last`` or ``min``, then the antialiased line pass will be
    performed in two stages.
    """

    def __init__(self, **kwargs):
        ks, vs = zip(*sorted(kwargs.items()))
        self.keys = ks
        self.values = vs

    def __hash__(self):
        return hash((type(self), tuple(self.keys), tuple(self.values)))

    def is_categorical(self):
        for v in self.values:
            if v.is_categorical():
                return True
        return False

    def uses_row_index(self, cuda, partitioned):
        for v in self.values:
            if v.uses_row_index(cuda, partitioned):
                return True
        return False

    def validate(self, input_dshape):
        for v in self.values:
            v.validate(input_dshape)
        n_values = []
        for v in self.values:
            if isinstance(v, where):
                v = v.selector
            if isinstance(v, FloatingNReduction):
                n_values.append(v.n)
        if len(np.unique(n_values)) > 1:
            raise ValueError('Using multiple FloatingNReductions with different n values is not supported')

    @property
    def inputs(self):
        return tuple(unique(concat((v.inputs for v in self.values))))