from datetime import datetime
from typing import (
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.frame import DataFrame
from triad.utils.assertion import assert_or_throw
from triad.utils.pyarrow import (
def to_schema(self, df: T) -> pa.Schema:
    """Extract pandas dataframe schema as pyarrow schema. This is a replacement
        of pyarrow.Schema.from_pandas, and it can correctly handle string type and
        empty dataframes

        :param df: pandas dataframe
        :raises ValueError: if pandas dataframe does not have named schema
        :return: pyarrow.Schema

        :Notice:
        The dataframe must be either empty, or with type pd.RangeIndex, pd.Int64Index
        or pd.UInt64Index and without a name, otherwise, `ValueError` will raise.
        """
    self.ensure_compatible(df)
    assert_or_throw(df.columns.dtype == 'object', ValueError('Pandas dataframe must have named schema'))

    def get_fields() -> Iterable[pa.Field]:
        if isinstance(df, pd.DataFrame) and len(df.index) > 0:
            yield from pa.Schema.from_pandas(df, preserve_index=False)
        else:
            for i in range(df.shape[1]):
                tp = df.dtypes.iloc[i]
                if tp == np.dtype('object') or pd.api.types.is_string_dtype(tp):
                    t = pa.string()
                elif isinstance(tp, pd.DatetimeTZDtype):
                    t = pa.timestamp(tp.unit, str(tp.tz))
                else:
                    t = to_pa_datatype(tp)
                yield pa.field(df.columns[i], t)
    fields: List[pa.Field] = []
    for field in get_fields():
        if pa.types.is_timestamp(field.type):
            fields.append(pa.field(field.name, pa.timestamp(TRIAD_DEFAULT_TIMESTAMP_UNIT, field.type.tz)))
        elif pa.types.is_large_string(field.type):
            fields.append(pa.field(field.name, pa.string()))
        else:
            fields.append(field)
    return pa.schema(fields)