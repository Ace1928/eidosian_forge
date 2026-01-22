import numpy as np
import shapely
def _path_from_polygon(polygon):
    from matplotlib.path import Path
    if isinstance(polygon, shapely.MultiPolygon):
        return Path.make_compound_path(*[_path_from_polygon(poly) for poly in polygon.geoms])
    else:
        return Path.make_compound_path(Path(np.asarray(polygon.exterior.coords)[:, :2]), *[Path(np.asarray(ring.coords)[:, :2]) for ring in polygon.interiors])