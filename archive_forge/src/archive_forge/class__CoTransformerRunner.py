from typing import List, no_type_check
from triad import ParamDict, Schema, SerializableRLock, assert_or_throw
from triad.utils.convert import to_type
from fugue.collections.partition import PartitionCursor
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.array_dataframe import ArrayDataFrame
from fugue.dataframe.utils import _df_eq
from fugue.exceptions import FugueWorkflowError
from fugue.execution.execution_engine import (
from fugue.rpc import EmptyRPCHandler, to_rpc_handler
from ..outputter import Outputter
from ..transformer.convert import _to_output_transformer
from ..transformer.transformer import CoTransformer, Transformer
class _CoTransformerRunner(object):

    def __init__(self, df: DataFrame, transformer: CoTransformer, ignore_errors: List[type]):
        self.schema = df.schema
        self.transformer = transformer
        self.ignore_errors = tuple(ignore_errors)

    def run(self, cursor: PartitionCursor, dfs: DataFrames) -> LocalDataFrame:
        self.transformer._cursor = cursor
        try:
            self.transformer.transform(dfs).as_local_bounded()
            return ArrayDataFrame([], self.transformer.output_schema)
        except self.ignore_errors:
            return ArrayDataFrame([], self.transformer.output_schema)

    def on_init(self, partition_no: int, dfs: DataFrames) -> None:
        s = self.transformer.partition_spec
        self.transformer._cursor = s.get_cursor(self.schema, partition_no)
        self.transformer.on_init(dfs)