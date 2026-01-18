from typing import Any, List, Type, no_type_check
from triad.collections import ParamDict
from triad.collections.schema import Schema
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_type
from fugue.collections.partition import PartitionCursor
from fugue.column import ColumnExpr
from fugue.column import SelectColumns as ColumnsSelect
from fugue.dataframe import ArrayDataFrame, DataFrame, DataFrames, LocalDataFrame
from fugue.exceptions import FugueWorkflowError
from fugue.execution import make_sql_engine
from fugue.execution.execution_engine import (
from fugue.extensions.processor import Processor
from fugue.extensions.transformer import CoTransformer, Transformer, _to_transformer
from fugue.rpc import EmptyRPCHandler, to_rpc_handler
def validate_on_compile(self):
    n = self.params.get_or_none('n', int)
    frac = self.params.get_or_none('frac', float)
    assert_or_throw(n is None and frac is not None or (n is not None and frac is None), ValueError('one and only one of n and frac should be set'))