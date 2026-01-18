import numpy as np
import pandas
import pyarrow as pa
import pytest
import modin.pandas as pd
from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
from modin.pandas.io import from_arrow, from_dataframe
from modin.tests.pandas.utils import df_equals
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import export_frame, get_data_of_all_types, split_df_into_chunks
@pytest.mark.parametrize('data_has_nulls', [True, False])
def test_simple_import(data_has_nulls):
    """Test that ``modin.pandas.utils.from_dataframe`` works properly."""
    data = get_data_of_all_types(data_has_nulls)
    modin_df_producer = pd.DataFrame(data)
    internal_modin_df_producer = modin_df_producer.__dataframe__()
    with warns_that_defaulting_to_pandas():
        modin_df_consumer = from_dataframe(modin_df_producer)
        internal_modin_df_consumer = from_dataframe(internal_modin_df_producer)
    assert modin_df_producer is not modin_df_consumer
    assert internal_modin_df_producer is not internal_modin_df_consumer
    assert modin_df_producer._query_compiler._modin_frame is not modin_df_consumer._query_compiler._modin_frame
    df_equals(modin_df_producer, modin_df_consumer)
    df_equals(modin_df_producer, internal_modin_df_consumer)