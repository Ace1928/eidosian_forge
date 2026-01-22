from toolz import memoize
import numpy as np
from datashader.glyphs.line import _build_map_onto_pixel_for_line
from datashader.glyphs.points import _GeometryLike
from datashader.utils import ngjit
def _build_draw_polygon(append, map_onto_pixel, x_mapper, y_mapper, expand_aggs_and_cols):

    @ngjit
    @expand_aggs_and_cols
    def draw_polygon(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, offsets, offset_multiplier, values, xs, ys, yincreasing, eligible, *aggs_and_cols):
        """Draw a polygon using a winding-number scan-line algorithm
        """
        xs.fill(np.nan)
        ys.fill(np.nan)
        yincreasing.fill(0)
        eligible.fill(1)
        start_index = offset_multiplier * offsets[0]
        stop_index = offset_multiplier * offsets[-1]
        poly_xmin = np.min(values[start_index:stop_index:2])
        poly_ymin = np.min(values[start_index + 1:stop_index:2])
        poly_xmax = np.max(values[start_index:stop_index:2])
        poly_ymax = np.max(values[start_index + 1:stop_index:2])
        if poly_xmax < xmin or poly_xmin > xmax or poly_ymax < ymin or (poly_ymin > ymax):
            return
        startxi, startyi = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, max(poly_xmin, xmin), max(poly_ymin, ymin))
        stopxi, stopyi = map_onto_pixel(sx, tx, sy, ty, xmin, xmax, ymin, ymax, min(poly_xmax, xmax), min(poly_ymax, ymax))
        stopxi += 1
        stopyi += 1
        if stopxi - startxi == 1 and stopyi - startyi == 1:
            append(i, startxi, startyi, *aggs_and_cols)
            return
        elif stopxi - startxi == 1:
            for yi in range(min(startyi, stopyi) + 1, max(startyi, stopyi)):
                append(i, startxi, yi, *aggs_and_cols)
            return
        elif stopyi - startyi == 1:
            for xi in range(min(startxi, stopxi) + 1, max(startxi, stopxi)):
                append(i, xi, startyi, *aggs_and_cols)
            return
        ei = 0
        for j in range(len(offsets) - 1):
            start = offset_multiplier * offsets[j]
            stop = offset_multiplier * offsets[j + 1]
            for k in range(start, stop - 2, 2):
                x0 = values[k]
                y0 = values[k + 1]
                x1 = values[k + 2]
                y1 = values[k + 3]
                x0c = x_mapper(x0) * sx + tx - 0.5
                y0c = y_mapper(y0) * sy + ty - 0.5
                x1c = x_mapper(x1) * sx + tx - 0.5
                y1c = y_mapper(y1) * sy + ty - 0.5
                if y1c > y0c:
                    xs[ei, 0] = x0c
                    ys[ei, 0] = y0c
                    xs[ei, 1] = x1c
                    ys[ei, 1] = y1c
                    yincreasing[ei] = 1
                elif y1c < y0c:
                    xs[ei, 1] = x0c
                    ys[ei, 1] = y0c
                    xs[ei, 0] = x1c
                    ys[ei, 0] = y1c
                    yincreasing[ei] = -1
                else:
                    continue
                ei += 1
        num_edges = ei
        for yi in range(startyi, stopyi):
            eligible.fill(1)
            for xi in range(startxi, stopxi):
                winding_number = 0
                for ei in range(num_edges):
                    if eligible[ei] == 0:
                        continue
                    x0c = xs[ei, 0]
                    x1c = xs[ei, 1]
                    y0c = ys[ei, 0]
                    y1c = ys[ei, 1]
                    if y0c >= yi or y1c < yi or (x0c < xi and x1c < xi):
                        eligible[ei] = 0
                        continue
                    if xi <= x0c and xi <= x1c:
                        winding_number += yincreasing[ei]
                    else:
                        ax = x0c - xi
                        ay = y0c - yi
                        bx = x1c - xi
                        by = y1c - yi
                        bxa = bx * ay - by * ax
                        if bxa < 0 or (bxa == 0 and yincreasing[ei]):
                            winding_number += yincreasing[ei]
                        else:
                            eligible[ei] = 0
                            continue
                if winding_number != 0:
                    append(i, xi, yi, *aggs_and_cols)
    return draw_polygon