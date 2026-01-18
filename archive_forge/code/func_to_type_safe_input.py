import pickle
from typing import Any, Iterable, List, Tuple, Optional
import pandas as pd
import pyarrow as pa
import pyspark
import pyspark.sql as ps
import pyspark.sql.types as pt
from packaging import version
from pyarrow.types import is_list, is_struct, is_timestamp
from pyspark.sql.pandas.types import (
from triad.collections import Schema
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP, cast_pa_table
from triad.utils.schema import quote_name
import fugue.api as fa
from fugue import DataFrame
from .misc import is_spark_dataframe
def to_type_safe_input(rows: Iterable[ps.Row], schema: Schema) -> Iterable[List[Any]]:
    struct_idx = [p for p, t in enumerate(schema.types) if pa.types.is_struct(t)]
    complex_list_idx = [p for p, t in enumerate(schema.types) if pa.types.is_list(t) and pa.types.is_nested(t.value_type)]
    if len(struct_idx) == 0 and len(complex_list_idx) == 0:
        for row in rows:
            yield list(row)
    elif len(complex_list_idx) == 0:
        for row in rows:
            r = list(row)
            for i in struct_idx:
                if r[i] is not None:
                    r[i] = r[i].asDict(recursive=True)
            yield r
    else:
        for row in rows:
            data = row.asDict(recursive=True)
            r = [data[n] for n in schema.names]
            yield r