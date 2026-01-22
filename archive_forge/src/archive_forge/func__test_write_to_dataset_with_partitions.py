import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def _test_write_to_dataset_with_partitions(base_path, filesystem=None, schema=None, index_name=None):
    import pandas as pd
    import pandas.testing as tm
    import pyarrow.parquet as pq
    output_df = pd.DataFrame({'group1': list('aaabbbbccc'), 'group2': list('eefeffgeee'), 'num': list(range(10)), 'nan': [np.nan] * 10, 'date': np.arange('2017-01-01', '2017-01-11', dtype='datetime64[D]').astype('datetime64[ns]')})
    cols = output_df.columns.tolist()
    partition_by = ['group1', 'group2']
    output_table = pa.Table.from_pandas(output_df, schema=schema, safe=False, preserve_index=False)
    pq.write_to_dataset(output_table, base_path, partition_by, filesystem=filesystem)
    metadata_path = os.path.join(str(base_path), '_common_metadata')
    if filesystem is not None:
        with filesystem.open(metadata_path, 'wb') as f:
            pq.write_metadata(output_table.schema, f)
    else:
        pq.write_metadata(output_table.schema, metadata_path)
    dataset = pq.ParquetDataset(base_path, filesystem=filesystem)
    dataset_cols = set(dataset.schema.names)
    assert dataset_cols == set(output_table.schema.names)
    input_table = dataset.read()
    input_df = input_table.to_pandas()
    input_df_cols = input_df.columns.tolist()
    assert partition_by == input_df_cols[-1 * len(partition_by):]
    input_df = input_df[cols]
    for col in partition_by:
        output_df[col] = output_df[col].astype('category')
    if schema:
        expected_date_type = schema.field('date').type.to_pandas_dtype()
        output_df['date'] = output_df['date'].astype(expected_date_type)
    tm.assert_frame_equal(output_df, input_df)