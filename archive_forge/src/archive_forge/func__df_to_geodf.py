import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def _df_to_geodf(df, geom_col='geom', crs=None):
    """
    Transforms a pandas DataFrame into a GeoDataFrame.
    The column 'geom_col' must be a geometry column in WKB representation.
    To be used to convert df based on pd.read_sql to gdf.
    Parameters
    ----------
    df : DataFrame
        pandas DataFrame with geometry column in WKB representation.
    geom_col : string, default 'geom'
        column name to convert to shapely geometries
    crs : pyproj.CRS, optional
        CRS to use for the returned GeoDataFrame. The value can be anything accepted
        by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
        If not set, tries to determine CRS from the SRID associated with the
        first geometry in the database, and assigns that to all geometries.
    Returns
    -------
    GeoDataFrame
    """
    if geom_col not in df:
        raise ValueError("Query missing geometry column '{}'".format(geom_col))
    if df.columns.to_list().count(geom_col) > 1:
        raise ValueError(f"Duplicate geometry column '{geom_col}' detected in SQL query output. Onlyone geometry column is allowed.")
    geoms = df[geom_col].dropna()
    if not geoms.empty:
        load_geom_bytes = shapely.wkb.loads
        'Load from Python 3 binary.'

        def load_geom_buffer(x):
            """Load from Python 2 binary."""
            return shapely.wkb.loads(str(x))

        def load_geom_text(x):
            """Load from binary encoded as text."""
            return shapely.wkb.loads(str(x), hex=True)
        if isinstance(geoms.iat[0], bytes):
            load_geom = load_geom_bytes
        else:
            load_geom = load_geom_text
        df[geom_col] = geoms = geoms.apply(load_geom)
        if crs is None:
            if compat.SHAPELY_GE_20:
                srid = shapely.get_srid(geoms.iat[0])
            else:
                srid = shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom)
            if srid != 0:
                crs = 'epsg:{}'.format(srid)
    return GeoDataFrame(df, crs=crs, geometry=geom_col)