import sys
from collections import defaultdict
from typing import (
from uuid import uuid4
from adagio.specs import WorkflowSpec
from triad import (
from fugue._utils.exception import modify_traceback
from fugue.collections.partition import PartitionSpec
from fugue.collections.sql import StructuredRawSQL
from fugue.collections.yielded import Yielded
from fugue.column import ColumnExpr
from fugue.column import SelectColumns as ColumnsSelect
from fugue.column import all_cols, col, lit
from fugue.constants import (
from fugue.dataframe import DataFrame, LocalBoundedDataFrame, YieldedDataFrame
from fugue.dataframe.api import is_df
from fugue.dataframe.dataframes import DataFrames
from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowError
from fugue.execution.api import engine_context
from fugue.extensions._builtins import (
from fugue.extensions.transformer.convert import _to_output_transformer, _to_transformer
from fugue.rpc import to_rpc_handler
from fugue.rpc.base import EmptyRPCHandler
from fugue.workflow._checkpoint import StrongCheckpoint, WeakCheckpoint
from fugue.workflow._tasks import Create, FugueTask, Output, Process
from fugue.workflow._workflow_context import FugueWorkflowContext
def yield_dataframe_as(self: TDF, name: str, as_local: bool=False) -> None:
    """Yield a dataframe that can be accessed without
        the current execution engine

        :param name: the name of the yielded dataframe
        :param as_local: yield the local version of the dataframe

        .. note::

            When ``as_local`` is True, it can trigger an additional compute
            to do the conversion. To avoid recompute, you should add
            ``persist`` before yielding.
        """
    yielded = YieldedDataFrame(self._task.__uuid__())
    self.workflow._yields[name] = yielded
    self._task.set_yield_dataframe_handler(lambda df: yielded.set_value(df), as_local=as_local)