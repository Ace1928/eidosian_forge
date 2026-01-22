import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
@memoize
def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
    x_name = self.x
    y_name = self.y
    name = self.name

    @ngjit
    @self.expand_aggs_and_cols(append)
    def perform_extend(i, j, plot_height, plot_width, xs, ys, xverts, yverts, yincreasing, eligible, intersect, *aggs_and_cols):
        xverts[0] = xs[j, i]
        xverts[1] = xs[j, i + 1]
        xverts[2] = xs[j + 1, i + 1]
        xverts[3] = xs[j + 1, i]
        xverts[4] = xverts[0]
        yverts[0] = ys[j, i]
        yverts[1] = ys[j, i + 1]
        yverts[2] = ys[j + 1, i + 1]
        yverts[3] = ys[j + 1, i]
        yverts[4] = yverts[0]
        xmin = min(min(xverts[0], xverts[1]), min(xverts[2], xverts[3]))
        if xmin >= plot_width:
            return
        xmin = max(xmin, 0)
        xmax = max(max(xverts[0], xverts[1]), max(xverts[2], xverts[3]))
        if xmax < 0:
            return
        xmax = min(xmax, plot_width)
        ymin = min(min(yverts[0], yverts[1]), min(yverts[2], yverts[3]))
        if ymin >= plot_height:
            return
        ymin = max(ymin, 0)
        ymax = max(max(yverts[0], yverts[1]), max(yverts[2], yverts[3]))
        if ymax < 0:
            return
        ymax = min(ymax, plot_height)
        if xmin == xmax or ymin == ymax:
            if xmin == xmax and xmax < plot_width:
                xmax += 1
            if ymin == ymax and ymax < plot_height:
                ymax += 1
            for yi in range(ymin, ymax):
                for xi in range(xmin, xmax):
                    append(j, i, xi, yi, *aggs_and_cols)
            return
        yincreasing[:] = 0
        for k in range(4):
            if yverts[k + 1] > yverts[k]:
                yincreasing[k] = 1
            elif yverts[k + 1] < yverts[k]:
                yincreasing[k] = -1
        for yi in range(ymin, ymax):
            eligible[:] = 1
            for xi in range(xmin, xmax):
                intersect[:] = 0
                for edge_i in range(4):
                    if not eligible[edge_i]:
                        continue
                    if xverts[edge_i] < xi and xverts[edge_i + 1] < xi:
                        eligible[edge_i] = 0
                        continue
                    if (yverts[edge_i] > yi) == (yverts[edge_i + 1] > yi):
                        eligible[edge_i] = 0
                        continue
                    ax = xverts[edge_i] - xi
                    ay = yverts[edge_i] - yi
                    bx = xverts[edge_i + 1] - xi
                    by = yverts[edge_i + 1] - yi
                    bxa = bx * ay - by * ax
                    intersect[edge_i] = bxa * yincreasing[edge_i] < 0
                intersections = intersect[0] + intersect[1] + intersect[2] + intersect[3]
                if intersections % 2 == 1:
                    append(j, i, xi, yi, *aggs_and_cols)

    @cuda.jit
    @self.expand_aggs_and_cols(append)
    def extend_cuda(plot_height, plot_width, xs, ys, *aggs_and_cols):
        xverts = cuda.local.array(5, dtype=numba.types.int32)
        yverts = cuda.local.array(5, dtype=numba.types.int32)
        yincreasing = cuda.local.array(4, dtype=numba.types.int8)
        eligible = cuda.local.array(4, dtype=numba.types.int8)
        intersect = cuda.local.array(4, dtype=numba.types.int8)
        i, j = cuda.grid(2)
        if i < xs.shape[0] - 1 and j < ys.shape[0] - 1:
            perform_extend(i, j, plot_height, plot_width, xs, ys, xverts, yverts, yincreasing, eligible, intersect, *aggs_and_cols)

    @ngjit
    @self.expand_aggs_and_cols(append)
    def extend_cpu(plot_height, plot_width, xs, ys, *aggs_and_cols):
        xverts = np.zeros(5, dtype=np.int32)
        yverts = np.zeros(5, dtype=np.int32)
        yincreasing = np.zeros(4, dtype=np.int8)
        eligible = np.ones(4, dtype=np.int8)
        intersect = np.zeros(4, dtype=np.int8)
        y_len, x_len = xs.shape
        for i in range(x_len - 1):
            for j in range(y_len - 1):
                perform_extend(i, j, plot_height, plot_width, xs, ys, xverts, yverts, yincreasing, eligible, intersect, *aggs_and_cols)

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
        plot_height, plot_width = aggs[0].shape[:2]
        xs = (xscaled * plot_width).astype(int)
        ys = (yscaled * plot_height).astype(int)
        coord_dims = xr_ds.coords[x_name].dims
        aggs_and_cols = aggs + info(xr_ds.transpose(*coord_dims), aggs[0].shape[:2])
        if use_cuda:
            do_extend = extend_cuda[cuda_args(xr_ds[name].shape)]
        else:
            do_extend = extend_cpu
        do_extend(plot_height, plot_width, xs, ys, *aggs_and_cols)
    return extend