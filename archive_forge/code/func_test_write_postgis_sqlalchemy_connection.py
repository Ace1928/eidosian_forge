import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_sqlalchemy_connection(self, engine_postgis, df_nybb):
    """Tests that GeoDataFrame can be written to PostGIS with defaults."""
    with engine_postgis.begin() as con:
        table = 'nybb_con'
        drop_table_if_exists(con, table)
        write_postgis(df_nybb, con=con, name=table, if_exists='fail')
        sql = text('SELECT * FROM {table};'.format(table=table))
        df = read_postgis(sql, con, geom_col='geometry')
        validate_boro_df(df)