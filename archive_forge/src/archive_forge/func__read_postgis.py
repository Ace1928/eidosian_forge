import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def _read_postgis(sql, con, geom_col='geom', crs=None, index_col=None, coerce_float=True, parse_dates=None, params=None, chunksize=None):
    """
    Returns a GeoDataFrame corresponding to the result of the query
    string, which must contain a geometry column in WKB representation.

    It is also possible to use :meth:`~GeoDataFrame.read_file` to read from a database.
    Especially for file geodatabases like GeoPackage or SpatiaLite this can be easier.

    Parameters
    ----------
    sql : string
        SQL query to execute in selecting entries from database, or name
        of the table to read from the database.
    con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Active connection to the database to query.
    geom_col : string, default 'geom'
        column name to convert to shapely geometries
    crs : dict or str, optional
        CRS to use for the returned GeoDataFrame; if not set, tries to
        determine CRS from the SRID associated with the first geometry in
        the database, and assigns that to all geometries.
    chunksize : int, default None
        If specified, return an iterator where chunksize is the number of rows to
        include in each chunk.

    See the documentation for pandas.read_sql for further explanation
    of the following parameters:
    index_col, coerce_float, parse_dates, params, chunksize

    Returns
    -------
    GeoDataFrame

    Examples
    --------
    PostGIS

    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> db_connection_url = "postgresql://myusername:mypassword@myhost:5432/mydatabase"
    >>> con = create_engine(db_connection_url)  # doctest: +SKIP
    >>> sql = "SELECT geom, highway FROM roads"
    >>> df = geopandas.read_postgis(sql, con)  # doctest: +SKIP

    SpatiaLite

    >>> sql = "SELECT ST_AsBinary(geom) AS geom, highway FROM roads"
    >>> df = geopandas.read_postgis(sql, con)  # doctest: +SKIP
    """
    if chunksize is None:
        df = pd.read_sql(sql, con, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, params=params, chunksize=chunksize)
        return _df_to_geodf(df, geom_col=geom_col, crs=crs)
    else:
        df_generator = pd.read_sql(sql, con, index_col=index_col, coerce_float=coerce_float, parse_dates=parse_dates, params=params, chunksize=chunksize)
        return (_df_to_geodf(df, geom_col=geom_col, crs=crs) for df in df_generator)