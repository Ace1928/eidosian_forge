from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
class AreaToZeroAxis1(_PointLike):

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        elif not all([isreal(in_dshape.measure[str(ycol)]) for ycol in self.y]):
            raise ValueError('y columns must be real')
        unique_x_measures = set((in_dshape.measure[str(xcol)] for xcol in self.x))
        if len(unique_x_measures) > 1:
            raise ValueError('x columns must have the same data type')
        unique_y_measures = set((in_dshape.measure[str(ycol)] for ycol in self.y))
        if len(unique_y_measures) > 1:
            raise ValueError('y columns must have the same data type')

    def required_columns(self):
        return self.x + self.y

    @property
    def x_label(self):
        return 'x'

    @property
    def y_label(self):
        return 'y'

    def compute_x_bounds(self, df):
        xs = tuple((df[xlabel] for xlabel in self.x))
        bounds_list = [self._compute_bounds(xcol) for xcol in xs]
        mins, maxes = zip(*bounds_list)
        return self.maybe_expand_bounds((min(mins), max(maxes)))

    def compute_y_bounds(self, df):
        ys = tuple((df[ylabel] for ylabel in self.y))
        bounds_list = [self._compute_bounds(ycol) for ycol in ys]
        mins, maxes = zip(*bounds_list)
        mn = min(0, min(mins))
        mx = max(0, max(maxes))
        return self.maybe_expand_bounds((mn, mx))

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin([np.nanmin(df[c].values).item() for c in self.x]), np.nanmax([np.nanmax(df[c].values).item() for c in self.x]), np.nanmin([np.nanmin(df[c].values).item() for c in self.y]), np.nanmax([np.nanmax(df[c].values).item() for c in self.y])]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        y_extents = (np.nanmin(r[:, 2]), np.nanmax(r[:, 3]))
        y_extents = (min(0, y_extents[0]), max(0, y_extents[1]))
        return (self.maybe_expand_bounds(x_extents), self.maybe_expand_bounds(y_extents))

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_area_to_zero_axis1_none_constant(draw_trapezoid_y, expand_aggs_and_cols)
        x_names = self.x
        y_names = self.y

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, list(x_names))
                ys = self.to_cupy_array(df, list(y_names))
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = df.loc[:, list(y_names)].to_numpy()
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols)
        return extend