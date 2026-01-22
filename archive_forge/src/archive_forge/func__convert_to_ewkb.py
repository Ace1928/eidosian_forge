import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def _convert_to_ewkb(gdf, geom_name, srid):
    """Convert geometries to ewkb."""
    if compat.USE_SHAPELY_20:
        geoms = shapely.to_wkb(shapely.set_srid(gdf[geom_name].values._data, srid=srid), hex=True, include_srid=True)
    elif compat.USE_PYGEOS:
        from pygeos import set_srid, to_wkb
        geoms = to_wkb(set_srid(gdf[geom_name].values._data, srid=srid), hex=True, include_srid=True)
    else:
        from shapely.wkb import dumps
        geoms = [dumps(geom, srid=srid, hex=True) for geom in gdf[geom_name]]
    df = pd.DataFrame(gdf, copy=False)
    df[geom_name] = geoms
    return df