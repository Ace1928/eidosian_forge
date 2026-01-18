import os
import warnings
import pandas as pd
import geopandas
from geopandas import GeoDataFrame, read_file, read_postgis
import geopandas._compat as compat
from geopandas.io.sql import _get_conn as get_conn, _write_postgis as write_postgis
from geopandas.tests.util import create_postgis, create_spatialite, validate_boro_df
import pytest
def test_write_postgis_to_different_schema_when_table_exists(self, engine_postgis, df_nybb):
    """
        Tests writing data to alternative schema.
        """
    engine = engine_postgis
    table = 'nybb'
    schema_to_use = 'test'
    sql = text('CREATE SCHEMA IF NOT EXISTS {schema};'.format(schema=schema_to_use))
    with engine.begin() as conn:
        conn.execute(sql)
    try:
        write_postgis(df_nybb, con=engine, name=table, if_exists='fail', schema=schema_to_use)
        sql = text('SELECT * FROM {schema}.{table};'.format(schema=schema_to_use, table=table))
        df = read_postgis(sql, engine, geom_col='geometry')
        validate_boro_df(df)
    except ValueError:
        pass
    write_postgis(df_nybb, con=engine, name=table, if_exists='replace', schema=schema_to_use)
    sql = text('SELECT * FROM {schema}.{table};'.format(schema=schema_to_use, table=table))
    df = read_postgis(sql, engine, geom_col='geometry')
    validate_boro_df(df)