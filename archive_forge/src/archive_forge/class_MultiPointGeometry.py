from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit
from numba import cuda
class MultiPointGeometry(_GeometryLike):

    @property
    def geom_dtypes(self):
        from spatialpandas.geometry import PointDtype, MultiPointDtype
        return (PointDtype, MultiPointDtype)

    @memoize
    def _build_extend(self, x_mapper, y_mapper, info, append, _antialias_stage_2, _antialias_stage_2_funcs):
        geometry_name = self.geometry

        @ngjit
        @self.expand_aggs_and_cols(append)
        def _perform_extend_points(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols):
            x = values[j]
            y = values[j + 1]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                xx = int(x_mapper(x) * sx + tx)
                yy = int(y_mapper(y) * sy + ty)
                xi, yi = (xx - 1 if x == xmax else xx, yy - 1 if y == ymax else yy)
                append(i, xi, yi, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_point_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, eligible_inds, *aggs_and_cols):
            for i in eligible_inds:
                if missing[i] is True:
                    continue
                _perform_extend_points(i, 2 * i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols)

        @ngjit
        @self.expand_aggs_and_cols(append)
        def extend_multipoint_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets, eligible_inds, *aggs_and_cols):
            for i in eligible_inds:
                if missing[i] is True:
                    continue
                start = offsets[i]
                stop = offsets[i + 1]
                for j in range(start, stop, 2):
                    _perform_extend_points(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols)

        def extend(aggs, df, vt, bounds):
            from spatialpandas.geometry import PointArray
            aggs_and_cols = aggs + info(df, aggs[0].shape[:2])
            sx, tx, sy, ty = vt
            xmin, xmax, ymin, ymax = bounds
            geometry = df[geometry_name].array
            if geometry._sindex is not None:
                eligible_inds = geometry.sindex.intersects((xmin, ymin, xmax, ymax))
            else:
                eligible_inds = np.arange(0, len(geometry), dtype='uint32')
            missing = geometry.isna()
            if isinstance(geometry, PointArray):
                values = geometry.flat_values
                extend_point_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, eligible_inds, *aggs_and_cols)
            else:
                values = geometry.buffer_values
                offsets = geometry.buffer_offsets[0]
                extend_multipoint_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, missing, offsets, eligible_inds, *aggs_and_cols)
        return extend