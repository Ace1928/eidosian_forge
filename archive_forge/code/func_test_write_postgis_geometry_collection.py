import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_geometry_collection(self, engine_postgis, df_geom_collection):
    """
        Tests that writing a mix of different geometry types is possible.
        """
    engine = engine_postgis
    table = 'geomtype_tests'
    write_postgis(df_geom_collection, con=engine, name=table, if_exists='replace')
    sql = text('SELECT DISTINCT(GeometryType(geometry)) FROM {table} ORDER BY 1;'.format(table=table))
    with engine.connect() as conn:
        geom_type = conn.execute(sql).fetchone()[0]
    sql = text('SELECT * FROM {table};'.format(table=table))
    df = read_postgis(sql, engine, geom_col='geometry')
    assert geom_type.upper() == 'GEOMETRYCOLLECTION'
    assert df.geom_type.unique()[0] == 'GeometryCollection'