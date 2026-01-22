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
class where(FloatingReduction):
    """
    Returns values from a ``lookup_column`` corresponding to a ``selector``
    reduction that is applied to some other column.

    If ``lookup_column`` is ``None`` then it uses the index of the row in the
    DataFrame instead of a named column. This is returned as an int64
    aggregation with -1 used to denote no value.

    Examples
    --------
    >>> canvas.line(df, 'x', 'y', agg=ds.where(ds.max("value"), "other"))  # doctest: +SKIP

    This returns the values of the "other" column that correspond to the
    maximum of the "value" column in each bin.

    Parameters
    ----------
    selector: Reduction
        Reduction used to select the values of the ``lookup_column`` which are
        returned by this ``where`` reduction.

    lookup_column : str | None
        Column containing values that are returned from this ``where``
        reduction, or ``None`` to return row indexes instead.
    """

    def __init__(self, selector: Reduction, lookup_column: str | None=None):
        if not isinstance(selector, (first, first_n, last, last_n, max, max_n, min, min_n, _max_or_min_row_index, _max_n_or_min_n_row_index)):
            raise TypeError('selector can only be a first, first_n, last, last_n, max, max_n, min or min_n reduction')
        if lookup_column is None:
            lookup_column = SpecialColumn.RowIndex
        super().__init__(lookup_column)
        self.selector = selector
        self.columns = (selector.column, lookup_column)

    def __hash__(self):
        return hash((type(self), self._hashable_inputs(), self.selector))

    def is_where(self):
        return True

    def out_dshape(self, input_dshape, antialias, cuda, partitioned):
        if self.column == SpecialColumn.RowIndex:
            return dshape(ct.int64)
        else:
            return dshape(ct.float64)

    def uses_cuda_mutex(self) -> UsesCudaMutex:
        return UsesCudaMutex.Local

    def uses_row_index(self, cuda, partitioned):
        return self.column == SpecialColumn.RowIndex or self.selector.uses_row_index(cuda, partitioned)

    def validate(self, in_dshape):
        if self.column != SpecialColumn.RowIndex:
            super().validate(in_dshape)
        self.selector.validate(in_dshape)
        if self.column != SpecialColumn.RowIndex and self.column == self.selector.column:
            raise ValueError('where and its contained reduction cannot use the same column')

    def _antialias_stage_2(self, self_intersect, array_module) -> tuple[AntialiasStage2]:
        ret = self.selector._antialias_stage_2(self_intersect, array_module)
        if self.column == SpecialColumn.RowIndex:
            ret = (AntialiasStage2(combination=ret[0].combination, zero=-1, n_reduction=ret[0].n_reduction),)
        return ret

    @staticmethod
    @ngjit
    def _append(x, y, agg, field, update_index):
        if agg.ndim > 2:
            shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field
        return update_index

    @staticmethod
    @ngjit
    def _append_antialias(x, y, agg, field, aa_factor, prev_aa_factor, update_index):
        if agg.ndim > 2:
            shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_antialias_cuda(x, y, agg, field, aa_factor, prev_aa_factor, update_index):
        if agg.ndim > 2:
            cuda_shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field
        return update_index

    @staticmethod
    @nb_cuda.jit(device=True)
    def _append_cuda(x, y, agg, field, update_index):
        if agg.ndim > 2:
            cuda_shift_and_insert(agg[y, x], field, update_index)
        else:
            agg[y, x] = field
        return update_index

    def _build_append(self, dshape, schema, cuda, antialias, self_intersect):
        if cuda:
            if antialias:
                return self._append_antialias_cuda
            else:
                return self._append_cuda
        elif antialias:
            return self._append_antialias
        else:
            return self._append

    def _build_bases(self, cuda, partitioned):
        selector = self.selector
        if isinstance(selector, (_first_or_last, _first_n_or_last_n)) and selector.uses_row_index(cuda, partitioned):
            row_index_selector = selector._create_row_index_selector()
            if self.column == SpecialColumn.RowIndex:
                row_index_selector._nan_check_column = self.selector.column
                return row_index_selector._build_bases(cuda, partitioned)
            else:
                new_where = where(row_index_selector, self.column)
                new_where._nan_check_column = self.selector.column
                return row_index_selector._build_bases(cuda, partitioned) + new_where._build_bases(cuda, partitioned)
        else:
            return selector._build_bases(cuda, partitioned) + super()._build_bases(cuda, partitioned)

    def _combine_callback(self, cuda, partitioned, categorical):
        selector = self.selector
        is_n_reduction = isinstance(selector, FloatingNReduction)
        if cuda:
            append = selector._append_cuda
        else:
            append = selector._append
        invalid = isminus1 if self.selector.uses_row_index(cuda, partitioned) else isnull

        @ngjit
        def combine_cpu_2d(aggs, selector_aggs):
            ny, nx = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    value = selector_aggs[1][y, x]
                    if not invalid(value) and append(x, y, selector_aggs[0], value) >= 0:
                        aggs[0][y, x] = aggs[1][y, x]

        @ngjit
        def combine_cpu_3d(aggs, selector_aggs):
            ny, nx, ncat = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    for cat in range(ncat):
                        value = selector_aggs[1][y, x, cat]
                        if not invalid(value) and append(x, y, selector_aggs[0][:, :, cat], value) >= 0:
                            aggs[0][y, x, cat] = aggs[1][y, x, cat]

        @ngjit
        def combine_cpu_n_3d(aggs, selector_aggs):
            ny, nx, n = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    for i in range(n):
                        value = selector_aggs[1][y, x, i]
                        if invalid(value):
                            break
                        update_index = append(x, y, selector_aggs[0], value)
                        if update_index < 0:
                            break
                        shift_and_insert(aggs[0][y, x], aggs[1][y, x, i], update_index)

        @ngjit
        def combine_cpu_n_4d(aggs, selector_aggs):
            ny, nx, ncat, n = aggs[0].shape
            for y in range(ny):
                for x in range(nx):
                    for cat in range(ncat):
                        for i in range(n):
                            value = selector_aggs[1][y, x, cat, i]
                            if invalid(value):
                                break
                            update_index = append(x, y, selector_aggs[0][:, :, cat, :], value)
                            if update_index < 0:
                                break
                            shift_and_insert(aggs[0][y, x, cat], aggs[1][y, x, cat, i], update_index)

        @nb_cuda.jit
        def combine_cuda_2d(aggs, selector_aggs):
            ny, nx = aggs[0].shape
            x, y = nb_cuda.grid(2)
            if x < nx and y < ny:
                value = selector_aggs[1][y, x]
                if not invalid(value) and append(x, y, selector_aggs[0], value) >= 0:
                    aggs[0][y, x] = aggs[1][y, x]

        @nb_cuda.jit
        def combine_cuda_3d(aggs, selector_aggs):
            ny, nx, ncat = aggs[0].shape
            x, y, cat = nb_cuda.grid(3)
            if x < nx and y < ny and (cat < ncat):
                value = selector_aggs[1][y, x, cat]
                if not invalid(value) and append(x, y, selector_aggs[0][:, :, cat], value) >= 0:
                    aggs[0][y, x, cat] = aggs[1][y, x, cat]

        @nb_cuda.jit
        def combine_cuda_n_3d(aggs, selector_aggs):
            ny, nx, n = aggs[0].shape
            x, y = nb_cuda.grid(2)
            if x < nx and y < ny:
                for i in range(n):
                    value = selector_aggs[1][y, x, i]
                    if invalid(value):
                        break
                    update_index = append(x, y, selector_aggs[0], value)
                    if update_index < 0:
                        break
                    cuda_shift_and_insert(aggs[0][y, x], aggs[1][y, x, i], update_index)

        @nb_cuda.jit
        def combine_cuda_n_4d(aggs, selector_aggs):
            ny, nx, ncat, n = aggs[0].shape
            x, y, cat = nb_cuda.grid(3)
            if x < nx and y < ny and (cat < ncat):
                for i in range(n):
                    value = selector_aggs[1][y, x, cat, i]
                    if invalid(value):
                        break
                    update_index = append(x, y, selector_aggs[0][:, :, cat, :], value)
                    if update_index < 0:
                        break
                    cuda_shift_and_insert(aggs[0][y, x, cat], aggs[1][y, x, cat, i], update_index)
        if is_n_reduction:
            if cuda:
                return combine_cuda_n_4d if categorical else combine_cuda_n_3d
            else:
                return combine_cpu_n_4d if categorical else combine_cpu_n_3d
        elif cuda:
            return combine_cuda_3d if categorical else combine_cuda_2d
        else:
            return combine_cpu_3d if categorical else combine_cpu_2d

    def _build_combine(self, dshape, antialias, cuda, partitioned, categorical=False):
        combine = self._combine_callback(cuda, partitioned, categorical)

        def wrapped_combine(aggs, selector_aggs):
            if len(aggs) == 1:
                pass
            elif cuda:
                assert len(aggs) == 2
                is_n_reduction = isinstance(self.selector, FloatingNReduction)
                shape = aggs[0].shape[:-1] if is_n_reduction else aggs[0].shape
                combine[cuda_args(shape)](aggs, selector_aggs)
            else:
                for i in range(1, len(aggs)):
                    combine((aggs[0], aggs[i]), (selector_aggs[0], selector_aggs[i]))
            return (aggs[0], selector_aggs[0])
        return wrapped_combine

    def _build_combine_temps(self, cuda, partitioned):
        return (self.selector,)

    def _build_create(self, required_dshape):
        if isinstance(self.selector, FloatingNReduction):
            return lambda shape, array_module: super(where, self)._build_create(required_dshape)(shape + (self.selector.n,), array_module)
        else:
            return super()._build_create(required_dshape)

    def _build_finalize(self, dshape):
        if isinstance(self.selector, FloatingNReduction):
            add_finalize_kwargs = self.selector._add_finalize_kwargs
        else:
            add_finalize_kwargs = None

        def finalize(bases, cuda=False, **kwargs):
            if add_finalize_kwargs is not None:
                kwargs = add_finalize_kwargs(**kwargs)
            return xr.DataArray(bases[-1], **kwargs)
        return finalize