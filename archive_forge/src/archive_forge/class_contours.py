import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class contours(Operation):
    """
    Given a Image with a single channel, annotate it with contour
    lines for a given set of contour levels.

    The return is an NdOverlay with a Contours layer for each given
    level, overlaid on top of the input Image.
    """
    output_type = Overlay
    levels = param.ClassSelector(default=10, class_=(list, int), doc='\n        A list of scalar values used to specify the contour levels.')
    group = param.String(default='Level', doc='\n        The group assigned to the output contours.')
    filled = param.Boolean(default=False, doc='\n        Whether to generate filled contours')
    overlaid = param.Boolean(default=False, doc='\n        Whether to overlay the contour on the supplied Element.')
    _per_element = True

    def _process(self, element, key=None):
        try:
            from contourpy import FillType, LineType, __version__ as contourpy_version, contour_generator
        except ImportError:
            raise ImportError('contours operation requires contourpy.') from None
        xs = element.dimension_values(0, True, flat=False)
        ys = element.dimension_values(1, True, flat=False)
        zs = element.dimension_values(2, flat=False)
        if xs.shape[0] != zs.shape[0]:
            xs = xs[:-1] + np.diff(xs, axis=0) / 2.0
        if xs.shape[1] != zs.shape[1]:
            xs = xs[:, :-1] + np.diff(xs, axis=1) / 2.0
        if ys.shape[0] != zs.shape[0]:
            ys = ys[:-1] + np.diff(ys, axis=0) / 2.0
        if ys.shape[1] != zs.shape[1]:
            ys = ys[:, :-1] + np.diff(ys, axis=1) / 2.0
        data = (xs, ys, zs)
        data_is_datetime = tuple((isdatetime(arr) for k, arr in enumerate(data)))
        if any(data_is_datetime):
            if any(data_is_datetime[:2]) and self.p.filled:
                raise RuntimeError('Datetime spatial coordinates are not supported for filled contour calculations.')
            try:
                from matplotlib.dates import date2num, num2date
            except ImportError:
                raise ImportError('contours operation using datetimes requires matplotlib.') from None
            data = tuple((date2num(d) if is_datetime else d for d, is_datetime in zip(data, data_is_datetime)))
        xdim, ydim = element.dimensions('key', label=True)
        if self.p.filled:
            contour_type = Polygons
        else:
            contour_type = Contours
        vdims = element.vdims[:1]
        levels = self.p.levels
        zmin, zmax = element.range(2)
        if isinstance(levels, int):
            if zmin == zmax:
                contours = contour_type([], [xdim, ydim], vdims)
                return element * contours if self.p.overlaid else contours
            else:
                locator = MaxNLocator(levels + 1)
                levels = locator.tick_values(zmin, zmax)
        else:
            levels = np.array(levels)
        if data_is_datetime[2]:
            levels = date2num(levels)
        crange = (levels.min(), levels.max())
        if self.p.filled:
            vdims = [vdims[0].clone(range=crange)]
        if Version(contourpy_version) >= Version('1.2'):
            line_type = LineType.ChunkCombinedNan
        else:
            line_type = LineType.ChunkCombinedOffset
        cont_gen = contour_generator(*data, line_type=line_type, fill_type=FillType.ChunkCombinedOffsetOffset)

        def coords_to_datetime(coords):
            nan_mask = np.isnan(coords)
            any_nan = np.any(nan_mask)
            if any_nan:
                coords[nan_mask] = 0
            coords = np.array(num2date(coords))
            if any_nan:
                coords[nan_mask] = np.nan
            return coords

        def points_to_datetime(points):
            xs, ys = np.split(points, 2, axis=1)
            if data_is_datetime[0]:
                xs = coords_to_datetime(xs)
            if data_is_datetime[1]:
                ys = coords_to_datetime(ys)
            return np.concatenate((xs, ys), axis=1)
        paths = []
        if self.p.filled:
            empty = np.array([[np.nan, np.nan]])
            for lower_level, upper_level in zip(levels[:-1], levels[1:]):
                filled = cont_gen.filled(lower_level, upper_level)
                if (points := filled[0][0]) is None:
                    continue
                exteriors = []
                interiors = []
                if any(data_is_datetime[0:2]):
                    points = points_to_datetime(points)
                offsets = filled[1][0]
                outer_offsets = filled[2][0]
                for jstart, jend in zip(outer_offsets[:-1], outer_offsets[1:]):
                    if exteriors:
                        exteriors.append(empty)
                    exteriors.append(points[offsets[jstart]:offsets[jstart + 1]])
                    interior = [points[offsets[j]:offsets[j + 1]] for j in range(jstart + 1, jend)]
                    interiors.append(interior)
                level = (lower_level + upper_level) / 2
                geom = {element.vdims[0].name: num2date(level) if data_is_datetime[2] else level, (xdim, ydim): np.concatenate(exteriors) if exteriors else []}
                if interiors:
                    geom['holes'] = interiors
                paths.append(geom)
        else:
            for level in levels:
                lines = cont_gen.lines(level)
                if (points := lines[0][0]) is None:
                    continue
                if any(data_is_datetime[0:2]):
                    points = points_to_datetime(points)
                if line_type == LineType.ChunkCombinedOffset:
                    offsets = lines[1][0]
                    if offsets is not None and len(offsets) > 2:
                        offsets = offsets[1:-1].astype(np.int64)
                        points = np.insert(points, offsets, np.nan, axis=0)
                geom = {element.vdims[0].name: num2date(level) if data_is_datetime[2] else level, (xdim, ydim): points if points is not None else []}
                paths.append(geom)
        contours = contour_type(paths, label=element.label, kdims=element.kdims, vdims=vdims)
        if self.p.overlaid:
            contours = element * contours
        return contours