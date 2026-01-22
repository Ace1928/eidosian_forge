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
class AssertNotEqual(Outputter):

    def process(self, dfs: DataFrames) -> None:
        assert_or_throw(len(dfs) > 1, FugueWorkflowError("can't accept single input"))
        expected = dfs[0]
        for i in range(1, len(dfs)):
            assert not _df_eq(expected, dfs[i], throw=False, **self.params)