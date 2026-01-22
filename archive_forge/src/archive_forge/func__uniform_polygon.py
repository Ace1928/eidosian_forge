from warnings import warn
import numpy
from shapely.geometry import MultiPoint
from geopandas.array import from_shapely, points_from_xy
from geopandas.geoseries import GeoSeries
def _uniform_polygon(geom, size, generator):
    """
    Sample uniformly from within a polygon using batched sampling.
    """
    xmin, ymin, xmax, ymax = geom.bounds
    candidates = []
    while len(candidates) < size:
        batch = points_from_xy(x=generator.uniform(xmin, xmax, size=size), y=generator.uniform(ymin, ymax, size=size))
        valid_samples = batch[batch.sindex.query(geom, predicate='contains')]
        candidates.extend(valid_samples)
    return GeoSeries(candidates[:size]).unary_union