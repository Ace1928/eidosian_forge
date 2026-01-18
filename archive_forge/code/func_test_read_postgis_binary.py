import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_read_postgis_binary(self, connection_spatialite, df_nybb):
    """Tests that geometry read as binary is accepted."""
    con = connection_spatialite
    geom_col = df_nybb.geometry.name
    create_spatialite(con, df_nybb)
    sql = 'SELECT ogc_fid, borocode, boroname, shape_leng, shape_area, ST_AsBinary("{0}") AS "{0}" FROM nybb'.format(geom_col)
    df = read_postgis(sql, con, geom_col=geom_col)
    validate_boro_df(df)