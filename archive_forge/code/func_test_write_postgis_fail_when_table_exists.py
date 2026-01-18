import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_fail_when_table_exists(self, engine_postgis, df_nybb):
    """
        Tests that uploading the same table raises error when: if_replace='fail'.
        """
    engine = engine_postgis
    table = 'nybb'
    write_postgis(df_nybb, con=engine, name=table, if_exists='replace')
    try:
        write_postgis(df_nybb, con=engine, name=table, if_exists='fail')
    except ValueError as e:
        if 'already exists' in str(e):
            pass
        else:
            raise e