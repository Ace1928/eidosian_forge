import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_read_postgis_get_srid(self, connection_postgis, df_nybb):
    """Tests that an SRID can be read from a geodatabase (GH #451)."""
    con = connection_postgis
    crs = 'epsg:4269'
    df_reproj = df_nybb.to_crs(crs)
    create_postgis(con, df_reproj, srid=4269)
    sql = 'SELECT * FROM nybb;'
    df = read_postgis(sql, con)
    validate_boro_df(df)
    assert df.crs == crs