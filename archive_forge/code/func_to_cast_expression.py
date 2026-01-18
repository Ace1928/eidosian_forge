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
def to_cast_expression(schema1: Any, schema2: Any, allow_name_mismatch: bool) -> Tuple[bool, List[str]]:
    schema1 = to_spark_schema(schema1)
    schema2 = to_spark_schema(schema2)
    assert_or_throw(len(schema1) == len(schema2), lambda: ValueError(f'schema mismatch: {schema1}, {schema2}'))
    expr: List[str] = []
    has_cast = False
    for i in range(len(schema1)):
        name_match = schema1[i].name == schema2[i].name
        assert_or_throw(name_match or allow_name_mismatch, lambda: ValueError(f'schema name mismatch: {schema1}, {schema2}'))
        n1, n2 = (quote_name(schema1[i].name, quote='`'), quote_name(schema2[i].name, quote='`'))
        if schema1[i].dataType != schema2[i].dataType:
            type2 = schema2[i].dataType.simpleString()
            if isinstance(schema1[i].dataType, pt.FractionalType) and isinstance(schema2[i].dataType, (pt.StringType, pt.IntegralType)):
                expr.append(f'CAST(IF(isnan({n1}) OR {n1} IS NULL, NULL, {n1}) AS {type2}) {n2}')
            else:
                expr.append(f'CAST({n1} AS {type2}) {n2}')
            has_cast = True
        elif schema1[i].name != schema2[i].name:
            expr.append(f'{n1} AS {n2}')
            has_cast = True
        else:
            expr.append(n1)
    return (has_cast, expr)