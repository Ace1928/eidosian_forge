import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_mixed_geometry_types(self, engine_postgis, df_mixed_single_and_multi):
    """
        Tests that writing a mix of single and MultiGeometries is possible.
        """
    engine = engine_postgis
    table = 'geomtype_tests'
    write_postgis(df_mixed_single_and_multi, con=engine, name=table, if_exists='replace')
    sql = text('SELECT DISTINCT GeometryType(geometry) FROM {table} ORDER BY 1;'.format(table=table))
    with engine.connect() as conn:
        res = conn.execute(sql).fetchall()
    assert res[0][0].upper() == 'LINESTRING'
    assert res[1][0].upper() == 'MULTILINESTRING'
    assert res[2][0].upper() == 'POINT'