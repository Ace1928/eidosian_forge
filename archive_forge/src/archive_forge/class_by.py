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
class by(Reduction):
    """Apply the provided reduction separately per category.

    Parameters
    ----------
    cats: str or CategoryPreprocess instance
        Name of column to aggregate over, or a categorizer object that returns categories.
        Resulting aggregate has an outer dimension axis along the categories present.
    reduction : Reduction
        Per-category reduction function.
    """

    def __init__(self, cat_column, reduction=count()):
        super().__init__()
        if isinstance(cat_column, CategoryPreprocess):
            self.categorizer = cat_column
        elif isinstance(cat_column, str):
            self.categorizer = category_codes(cat_column)
        else:
            raise TypeError('first argument must be a column name or a CategoryPreprocess instance')
        self.column = self.categorizer.column
        self.columns = (self.categorizer.column,)
        if (columns := getattr(reduction, 'columns', None)) is not None:
            self.columns += columns[::-1]
        else:
            self.columns += (getattr(reduction, 'column', None),)
        self.reduction = reduction
        if self.val_column is not None:
            self.preprocess = category_values(self.categorizer, self.val_column)
        else:
            self.preprocess = self.categorizer

    def __hash__(self):
        return hash((type(self), self._hashable_inputs(), self.categorizer._hashable_inputs(), self.reduction))

    def _build_temps(self, cuda=False):
        return tuple((by(self.categorizer, tmp) for tmp in self.reduction._build_temps(cuda)))

    @property
    def cat_column(self):
        return self.columns[0]

    @property
    def val_column(self):
        return self.columns[1]

    def validate(self, in_dshape):
        self.preprocess.validate(in_dshape)
        self.reduction.validate(in_dshape)

    def out_dshape(self, input_dshape, antialias, cuda, partitioned):
        cats = self.categorizer.categories(input_dshape)
        red_shape = self.reduction.out_dshape(input_dshape, antialias, cuda, partitioned)
        return dshape(Record([(c, red_shape) for c in cats]))

    @property
    def inputs(self):
        return (self.preprocess,)

    def is_categorical(self):
        return True

    def is_where(self):
        return self.reduction.is_where()

    @property
    def nan_check_column(self):
        return self.reduction.nan_check_column

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return self.reduction.uses_cuda_mutex()

    def uses_row_index(self, cuda, partitioned):
        return self.reduction.uses_row_index(cuda, partitioned)

    def _antialias_requires_2_stages(self):
        return self.reduction._antialias_requires_2_stages()

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        ret = self.reduction._antialias_stage_2(self_intersect, array_module)
        return (AntialiasStage2(combination=ret[0].combination, zero=ret[0].zero, n_reduction=ret[0].n_reduction, categorical=True),)

    def _build_create(self, required_dshape):
        n_cats = len(required_dshape.measure.fields)
        return lambda shape, array_module: self.reduction._build_create(required_dshape)(shape + (n_cats,), array_module)

    def _build_bases(self, cuda, partitioned):
        bases = self.reduction._build_bases(cuda, partitioned)
        if len(bases) == 1 and bases[0] is self:
            return bases
        return tuple((by(self.categorizer, base) for base in bases))

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        return self.reduction._build_append(dshape, schema, cuda, antialias, self_intersect)

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical=False):
        return self.reduction._build_combine(dshape, antialias, cuda, partitioned, True)

    def _build_combine_temps(self, cuda, partitioned):
        return self.reduction._build_combine_temps(cuda, partitioned)

    def _build_finalize(self, dshape):
        cats = list(self.categorizer.categories(dshape))

        def finalize(bases, cuda=False, **kwargs):
            kwargs = copy.deepcopy(kwargs)
            kwargs['dims'] += [self.cat_column]
            kwargs['coords'][self.cat_column] = cats
            return self.reduction._build_finalize(dshape)(bases, cuda=cuda, **kwargs)
        return finalize