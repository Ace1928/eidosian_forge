import io
import pandas
import pytest
import modin.pandas as pd
from modin.config import StorageFormat
from modin.tests.pandas.utils import default_to_pandas_ignore_string, df_equals
@pytest.mark.skipif(StorageFormat.get() != 'Hdk', reason='Lack of implementation for other storage formats.')
def test_string_cast():
    from modin.experimental.sql import query
    data = {'A': ['A', 'B', 'C'], 'B': ['A', 'B', 'C']}
    mdf = pd.DataFrame(data)
    pdf = pandas.DataFrame(data)
    df_equals(pdf, query('SELECT * FROM df', df=mdf))