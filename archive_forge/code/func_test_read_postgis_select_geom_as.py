import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_read_postgis_select_geom_as(self, connection_postgis, df_nybb):
    """Tests that a SELECT {geom} AS {some_other_geom} works."""
    con = connection_postgis
    orig_geom = 'geom'
    out_geom = 'the_geom'
    create_postgis(con, df_nybb, geom_col=orig_geom)
    sql = 'SELECT borocode, boroname, shape_leng, shape_area,\n                    {} as {} FROM nybb;'.format(orig_geom, out_geom)
    df = read_postgis(sql, con, geom_col=out_geom)
    validate_boro_df(df)