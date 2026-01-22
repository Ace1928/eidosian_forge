import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
class QuadMeshRectilinear(_QuadMeshLike):

    def _compute_bounds_from_1d_centers(self, xr_ds, dim, maybe_expand=False, orient=True):
        vals = xr_ds[dim].values
        v0, v1, v_nm1, v_n = [vals[i] for i in [0, 1, -2, -1]]
        if v_n < v0:
            descending = True
            v0, v1, v_nm1, v_n = (v_n, v_nm1, v1, v0)
        else:
            descending = False
        bounds = (v0 - 0.5 * (v1 - v0), v_n + 0.5 * (v_n - v_nm1))
        if not orient and descending:
            bounds = (bounds[1], bounds[0])
        if maybe_expand:
            bounds = self.maybe_expand_bounds(bounds)
        return bounds

    def compute_x_bounds(self, xr_ds):
        return self._compute_bounds_from_1d_centers(xr_ds, self.x, maybe_expand=True)

    def compute_y_bounds(self, xr_ds):
        return self._compute_bounds_from_1d_centers(xr_ds, self.y, maybe_expand=True)

    def compute_bounds_dask(self, xr_ds):
        return (self.compute_x_bounds(xr_ds), self.compute_y_bounds(xr_ds))

    def infer_interval_breaks(self, centers):
        return infer_interval_breaks(centers)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        x_name = self.x
        y_name = self.y
        name = self.name

        @ngjit
        @self.expand_aggs_and_cols(append)
        def perform_extend(i, j, xs, ys, *aggs_and_cols):
            x0i, x1i = (xs[i], xs[i + 1])
            if x0i > x1i:
                x0i, x1i = (x1i, x0i)
            if x0i == x1i:
                x1i += 1
            y0i, y1i = (ys[j], ys[j + 1])
            if y0i > y1i:
                y0i, y1i = (y1i, y0i)
            if y0i == y1i:
                y1i += 1
            for xi in range(x0i, x1i):
                for yi in range(y0i, y1i):
                    append(j, i, xi, yi, *aggs_and_cols)

        @cuda.jit
        @self.expand_aggs_and_cols(append)
        def extend_cuda(xs, ys, *aggs_and_cols):
            i, j = cuda.grid(2)
            if i < xs.shape[0] - 1 and j < ys.shape[0] - 1:
                perform_extend(i, j, xs, ys, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_cpu(xs, ys, *aggs_and_cols):
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    perform_extend(i, j, xs, ys, *aggs_and_cols)

        def extend(aggs, xr_ds, vt, bounds, x_breaks=None, y_breaks=None):
            from datashader.core import LinearAxis
            use_cuda = cupy and isinstance(xr_ds[name].data, cupy.ndarray)
            if use_cuda:
                x_mapper2 = _cuda_mapper(x_mapper)
                y_mapper2 = _cuda_mapper(y_mapper)
            else:
                x_mapper2 = x_mapper
                y_mapper2 = y_mapper
            if x_breaks is None:
                x_centers = xr_ds[x_name].values
                if use_cuda:
                    x_centers = cupy.array(x_centers)
                x_breaks = self.infer_interval_breaks(x_centers)
            if y_breaks is None:
                y_centers = xr_ds[y_name].values
                if use_cuda:
                    y_centers = cupy.array(y_centers)
                y_breaks = self.infer_interval_breaks(y_centers)
            x0, x1, y0, y1 = bounds
            xspan = x1 - x0
            yspan = y1 - y0
            if x_mapper is LinearAxis.mapper:
                xscaled = (x_breaks - x0) / xspan
            else:
                xscaled = (x_mapper2(x_breaks) - x0) / xspan
            if y_mapper is LinearAxis.mapper:
                yscaled = (y_breaks - y0) / yspan
            else:
                yscaled = (y_mapper2(y_breaks) - y0) / yspan
            xinds = np.where((xscaled >= 0) & (xscaled <= 1))[0]
            yinds = np.where((yscaled >= 0) & (yscaled <= 1))[0]
            if len(xinds) == 0 or len(yinds) == 0:
                return
            xm0, xm1 = (max(xinds.min() - 1, 0), xinds.max() + 1)
            ym0, ym1 = (max(yinds.min() - 1, 0), yinds.max() + 1)
            plot_height, plot_width = aggs[0].shape[:2]
            xs = (xscaled[xm0:xm1 + 1] * plot_width).astype(int).clip(0, plot_width)
            ys = (yscaled[ym0:ym1 + 1] * plot_height).astype(int).clip(0, plot_height)
            cols_full = info(xr_ds.transpose(y_name, x_name), aggs[0].shape[:2])
            cols = tuple([c[ym0:ym1, xm0:xm1] for c in cols_full])
            aggs_and_cols = aggs + cols
            if use_cuda:
                do_extend = extend_cuda[cuda_args(xr_ds[name].shape)]
            else:
                do_extend = extend_cpu
            do_extend(xs, ys, *aggs_and_cols)
        return extend