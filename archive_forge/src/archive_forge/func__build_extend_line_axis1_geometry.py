from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
def _build_extend_line_axis1_geometry(draw_segment, expand_aggs_and_cols, antialias_stage_2_funcs):
    if antialias_stage_2_funcs is not None:
        aa_stage_2_accumulate, aa_stage_2_clear, aa_stage_2_copy_back = antialias_stage_2_funcs
    use_2_stage_agg = antialias_stage_2_funcs is not None

    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geometry, closed_rings, antialias_stage_2, *aggs_and_cols):
        values = geometry.buffer_values
        missing = geometry.isna()
        offsets = geometry.buffer_offsets
        if len(offsets) == 2:
            offsets0, offsets1 = offsets
        else:
            offsets1 = offsets[0]
            offsets0 = np.arange(len(offsets1))
        if geometry._sindex is not None:
            eligible_inds = geometry.sindex.intersects((xmin, ymin, xmax, ymax))
        else:
            eligible_inds = np.arange(0, len(geometry), dtype='uint32')
        extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets0, offsets1, eligible_inds, closed_rings, antialias_stage_2, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets0, offsets1, eligible_inds, closed_rings, antialias_stage_2, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        for i in eligible_inds:
            if missing[i]:
                continue
            start0 = offsets0[i]
            stop0 = offsets0[i + 1]
            for j in range(start0, stop0):
                start1 = offsets1[j]
                stop1 = offsets1[j + 1]
                for k in range(start1, stop1 - 2, 2):
                    x0 = values[k]
                    if not np.isfinite(x0):
                        continue
                    y0 = values[k + 1]
                    if not np.isfinite(y0):
                        continue
                    x1 = values[k + 2]
                    if not np.isfinite(x1):
                        continue
                    y1 = values[k + 3]
                    if not np.isfinite(y1):
                        continue
                    segment_start = k == start1 and (not closed_rings) or (k > start1 and (not np.isfinite(values[k - 2]) or not np.isfinite(values[k - 1])))
                    segment_end = not closed_rings and k == stop1 - 4 or (k < stop1 - 4 and (not np.isfinite(values[k + 4]) or not np.isfinite(values[k + 5])))
                    if segment_start or use_2_stage_agg:
                        xm = 0.0
                        ym = 0.0
                    elif k == start1 and closed_rings:
                        xm = values[stop1 - 4]
                        ym = values[stop1 - 3]
                    else:
                        xm = values[k - 2]
                        ym = values[k - 1]
                    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, xm, ym, buffer, *aggs_and_cols)

    def extend_cpu_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, geometry, closed_rings, antialias_stage_2, *aggs_and_cols):
        values = geometry.buffer_values
        missing = geometry.isna()
        offsets = geometry.buffer_offsets
        if len(offsets) == 2:
            offsets0, offsets1 = offsets
        else:
            offsets1 = offsets[0]
            offsets0 = np.arange(len(offsets1))
        if geometry._sindex is not None:
            eligible_inds = geometry.sindex.intersects((xmin, ymin, xmax, ymax))
        else:
            eligible_inds = np.arange(0, len(geometry), dtype='uint32')
        n_aggs = len(antialias_stage_2[0])
        aggs_and_accums = tuple(((agg, agg.copy()) for agg in aggs_and_cols[:n_aggs]))
        extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets0, offsets1, eligible_inds, closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu_numba_antialias_2agg(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets0, offsets1, eligible_inds, closed_rings, antialias_stage_2, aggs_and_accums, *aggs_and_cols):
        antialias = antialias_stage_2 is not None
        buffer = np.empty(8) if antialias else None
        first_pass = True
        for i in eligible_inds:
            if missing[i]:
                continue
            start0 = offsets0[i]
            stop0 = offsets0[i + 1]
            for j in range(start0, stop0):
                start1 = offsets1[j]
                stop1 = offsets1[j + 1]
                for k in range(start1, stop1 - 2, 2):
                    x0 = values[k]
                    if not np.isfinite(x0):
                        continue
                    y0 = values[k + 1]
                    if not np.isfinite(y0):
                        continue
                    x1 = values[k + 2]
                    if not np.isfinite(x1):
                        continue
                    y1 = values[k + 3]
                    if not np.isfinite(y1):
                        continue
                    segment_start = k == start1 and (not closed_rings) or (k > start1 and (not np.isfinite(values[k - 2]) or not np.isfinite(values[k - 1])))
                    segment_end = not closed_rings and k == stop1 - 4 or (k < stop1 - 4 and (not np.isfinite(values[k + 4]) or not np.isfinite(values[k + 5])))
                    draw_segment(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, segment_start, segment_end, x0, x1, y0, y1, 0.0, 0.0, buffer, *aggs_and_cols)
            aa_stage_2_accumulate(aggs_and_accums, first_pass)
            first_pass = False
            aa_stage_2_clear(aggs_and_accums)
        aa_stage_2_copy_back(aggs_and_accums)
    if use_2_stage_agg:
        return extend_cpu_antialias_2agg
    else:
        return extend_cpu