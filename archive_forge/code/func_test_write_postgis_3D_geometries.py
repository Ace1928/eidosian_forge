import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_3D_geometries(self, engine_postgis, df_3D_geoms):
    """
        Tests writing a geometries with 3 dimensions works.
        """
    engine = engine_postgis
    table = 'geomtype_tests'
    write_postgis(df_3D_geoms, con=engine, name=table, if_exists='replace')
    sql = text('SELECT * FROM {table};'.format(table=table))
    df = read_postgis(sql, engine, geom_col='geometry')
    assert list(df.geometry.has_z) == [True, True, True]