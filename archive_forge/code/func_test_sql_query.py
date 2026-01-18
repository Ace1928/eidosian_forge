import io
import pandas
import pytest
import modin.pandas as pd
from modin.config import StorageFormat
from modin.tests.pandas.utils import default_to_pandas_ignore_string, df_equals
@pytest.mark.skipif(StorageFormat.get() != 'Hdk', reason='Lack of implementation for other storage formats.')
def test_sql_query():
    from modin.experimental.sql import query
    df = pd.read_csv(io.StringIO(titanic_snippet))
    sql = 'SELECT survived, p_class, count(passenger_id) as cnt FROM (SELECT * FROM titanic WHERE survived = 1) as t1 GROUP BY survived, p_class'
    query_result = query(sql, titanic=df)
    expected_df = df[df.survived == 1].groupby(['survived', 'p_class']).agg({'passenger_id': 'count'}).reset_index()
    assert query_result.shape == expected_df.shape
    values_left = expected_df.dropna().values
    values_right = query_result.dropna().values
    assert (values_left == values_right).all()