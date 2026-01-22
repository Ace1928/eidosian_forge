import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
class QuadMeshRaster(QuadMeshRectilinear):

    def is_upsample(self, source, x, y, name, x_range, y_range, out_w, out_h):
        src_w = len(source[x])
        if x_range is None:
            upsample_width = out_w >= src_w
        else:
            out_x0, out_x1 = x_range
            src_x0, src_x1 = self._compute_bounds_from_1d_centers(source, x, maybe_expand=False, orient=False)
            src_xbinsize = math.fabs((src_x1 - src_x0) / src_w)
            out_xbinsize = math.fabs((out_x1 - out_x0) / out_w)
            upsample_width = src_xbinsize >= out_xbinsize
        src_h = len(source[y])
        if y_range is None:
            upsample_height = out_h >= src_h
        else:
            out_y0, out_y1 = y_range
            src_y0, src_y1 = self._compute_bounds_from_1d_centers(source, y, maybe_expand=False, orient=False)
            src_ybinsize = math.fabs((src_y1 - src_y0) / src_h)
            out_ybinsize = math.fabs((out_y1 - out_y0) / out_h)
            upsample_height = src_ybinsize >= out_ybinsize
        return (upsample_width, upsample_height)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        x_name = self.x
        y_name = self.y
        name = self.name

        @ngjit_parallel
        def upsample_cpu(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, agg, col):
            for out_j in prange(out_h):
                src_j = int(math.floor(scale_y * (out_j + 0.5) + translate_y - offset_y))
                for out_i in range(out_w):
                    src_i = int(math.floor(scale_x * (out_i + 0.5) + translate_x - offset_x))
                    if src_j < 0 or src_j >= src_h or src_i < 0 or (src_i >= src_w):
                        agg[out_j, out_i] = np.nan
                    else:
                        agg[out_j, out_i] = col[src_j, src_i]

        @cuda.jit
        def upsample_cuda(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, agg, col):
            out_i, out_j = cuda.grid(2)
            if out_i < out_w and out_j < out_h:
                src_j = int(math.floor(scale_y * (out_j + 0.5) + translate_y - offset_y))
                src_i = int(math.floor(scale_x * (out_i + 0.5) + translate_x - offset_x))
                if src_j < 0 or src_j >= src_h or src_i < 0 or (src_i >= src_w):
                    agg[out_j, out_i] = np.nan
                else:
                    agg[out_j, out_i] = col[src_j, src_i]

        @ngjit_parallel
        @self.expand_aggs_and_cols(append)
        def downsample_cpu(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, *aggs_and_cols):
            for out_j in prange(out_h):
                src_j0 = int(max(math.floor(scale_y * (out_j + 0.0) + translate_y - offset_y), 0))
                src_j1 = int(min(math.floor(scale_y * (out_j + 1.0) + translate_y - offset_y), src_h))
                for out_i in range(out_w):
                    src_i0 = int(max(math.floor(scale_x * (out_i + 0.0) + translate_x - offset_x), 0))
                    src_i1 = int(min(math.floor(scale_x * (out_i + 1.0) + translate_x - offset_x), src_w))
                    for src_j in range(src_j0, src_j1):
                        for src_i in range(src_i0, src_i1):
                            append(src_j, src_i, out_i, out_j, *aggs_and_cols)

        @cuda.jit
        @self.expand_aggs_and_cols(append)
        def downsample_cuda(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, *aggs_and_cols):
            out_i, out_j = cuda.grid(2)
            if out_i < out_w and out_j < out_h:
                src_j0 = max(math.floor(scale_y * (out_j + 0.0) + translate_y - offset_y), 0)
                src_j1 = min(math.floor(scale_y * (out_j + 1.0) + translate_y - offset_y), src_h)
                src_i0 = max(math.floor(scale_x * (out_i + 0.0) + translate_x - offset_x), 0)
                src_i1 = min(math.floor(scale_x * (out_i + 1.0) + translate_x - offset_x), src_w)
                for src_j in range(src_j0, src_j1):
                    for src_i in range(src_i0, src_i1):
                        append(src_j, src_i, out_i, out_j, *aggs_and_cols)

        def extend(aggs, xr_ds, vt, bounds, scale_x=None, scale_y=None, translate_x=None, translate_y=None, offset_x=None, offset_y=None, src_xbinsize=None, src_ybinsize=None):
            use_cuda = cupy and isinstance(xr_ds[name].data, cupy.ndarray)
            out_h, out_w = aggs[0].shape
            out_x0, out_x1, out_y0, out_y1 = bounds
            out_xbinsize = math.fabs((out_x1 - out_x0) / out_w)
            out_ybinsize = math.fabs((out_y1 - out_y0) / out_h)
            xr_ds = xr_ds.transpose(y_name, x_name)
            src_h, src_w = xr_ds[name].shape
            if scale_x is None or scale_y is None or translate_x is None or (translate_y is None) or (offset_x is None) or (offset_y is None) or (src_xbinsize is None) or (src_ybinsize is None):
                src_x0, src_x1 = self._compute_bounds_from_1d_centers(xr_ds, x_name, maybe_expand=False, orient=False)
                src_y0, src_y1 = self._compute_bounds_from_1d_centers(xr_ds, y_name, maybe_expand=False, orient=False)
                src_xbinsize = math.fabs((src_x1 - src_x0) / src_w)
                src_ybinsize = math.fabs((src_y1 - src_y0) / src_h)
                scale_y, translate_y = build_scale_translate(out_h, out_y0, out_y1, src_h, src_y0, src_y1)
                scale_x, translate_x = build_scale_translate(out_w, out_x0, out_x1, src_w, src_x0, src_x1)
                offset_x = offset_y = 0
            cols = info(xr_ds, aggs[0].shape[:2])
            aggs_and_cols = tuple(aggs) + tuple(cols)
            if src_h == 0 or src_w == 0 or out_h == 0 or (out_w == 0):
                return
            elif src_xbinsize >= out_xbinsize and src_ybinsize >= out_ybinsize:
                if use_cuda:
                    do_sampling = upsample_cuda[cuda_args((out_w, out_h))]
                else:
                    do_sampling = upsample_cpu
                return do_sampling(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, aggs[0], cols[0])
            else:
                if use_cuda:
                    do_sampling = downsample_cuda[cuda_args((out_w, out_h))]
                else:
                    do_sampling = downsample_cpu
                return do_sampling(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, *aggs_and_cols)
        return extend