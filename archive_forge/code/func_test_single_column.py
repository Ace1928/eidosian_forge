from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
@pytest.mark.filterwarnings("ignore:The default dtype for empty Series will be 'object' instead of 'float64'")
@pytest.mark.parametrize('use_head', [True, False])
def test_single_column(db, use_head):
    from sqlalchemy import Column, Integer, MetaData, Table, create_engine
    with tmpfile() as f:
        uri = 'sqlite:///%s' % f
        metadata = MetaData()
        engine = create_engine(uri)
        table = Table('single_column', metadata, Column('id', Integer, primary_key=True))
        metadata.create_all(engine)
        test_data = pd.DataFrame({'id': list(range(50))}).set_index('id')
        test_data.to_sql(table.name, uri, index=True, if_exists='replace')
        if use_head:
            dask_df = read_sql_table(table.name, uri, index_col='id', npartitions=2)
        else:
            dask_df = read_sql_table(table.name, uri, head_rows=0, npartitions=2, meta=test_data.iloc[:0], index_col='id')
        assert dask_df.index.name == 'id'
        assert dask_df.npartitions == 2
        pd_dataframe = dask_df.compute()
        assert_eq(test_data, pd_dataframe)