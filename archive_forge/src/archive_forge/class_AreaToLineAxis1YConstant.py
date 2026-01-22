from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
class AreaToLineAxis1YConstant(AreaToLineAxis1):

    def validate(self, in_dshape):
        if not all([isreal(in_dshape.measure[str(xcol)]) for xcol in self.x]):
            raise ValueError('x columns must be real')
        unique_x_measures = set((in_dshape.measure[str(xcol)] for xcol in self.x))
        if len(unique_x_measures) > 1:
            raise ValueError('x columns must have the same data type')

    def required_columns(self):
        return self.x

    def compute_y_bounds(self, *args):
        y_min = min(np.nanmin(self.y), np.nanmin(self.y_stack))
        y_max = max(np.nanmax(self.y), np.nanmax(self.y_stack))
        return self.maybe_expand_bounds((y_min, y_max))

    @memoize
    def compute_bounds_dask(self, ddf):
        r = ddf.map_partitions(lambda df: np.array([[np.nanmin([np.nanmin(df[c].values).item() for c in self.x]), np.nanmax([np.nanmax(df[c].values).item() for c in self.x])]])).compute()
        x_extents = (np.nanmin(r[:, 0]), np.nanmax(r[:, 1]))
        return (self.maybe_expand_bounds(x_extents), self.compute_y_bounds())

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        expand_aggs_and_cols = self.expand_aggs_and_cols(append)
        map_onto_pixel = _build_map_onto_pixel_for_line(x_mapper, y_mapper)
        draw_trapezoid_y = _build_draw_trapezoid_y(append, map_onto_pixel, expand_aggs_and_cols)
        extend_cpu, extend_cuda = _build_extend_area_to_line_axis1_y_constant(draw_trapezoid_y, expand_aggs_and_cols)
        x_names = self.x
        y_values = self.y
        y_stack_values = self.y_stack

        def extend(aggs, df, vt, bounds, plot_start=True):
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            if cudf and isinstance(df, cudf.DataFrame):
                xs = self.to_cupy_array(df, list(x_names))
                ys = cp.asarray(y_values)
                ysv = cp.asarray(y_stack_values)
                do_extend = extend_cuda[cuda_args(xs.shape)]
            else:
                xs = df.loc[:, list(x_names)].to_numpy()
                ys = y_values
                ysv = y_stack_values
                do_extend = extend_cpu
            do_extend(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, ysv, *aggs_and_cols)
        return extend