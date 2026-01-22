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
class WorkflowDataFrames(DataFrames):
    """Ordered dictionary of WorkflowDataFrames. There are two modes: with keys
    and without keys. If without key ``_<n>`` will be used as the key
    for each dataframe, and it will be treated as an array in Fugue framework.

    It's immutable, once initialized, you can't add or remove element from it.

    It's a subclass of
    :class:`~fugue.dataframe.dataframes.DataFrames`, but different from
    DataFrames, in the initialization you should always use
    :class:`~fugue.workflow.workflow.WorkflowDataFrame`, and they should all
    come from the same :class:`~fugue.workflow.workflow.FugueWorkflow`.

    .. admonition:: Examples

        .. code-block:: python

            dag = FugueWorkflow()
            df1 = dag.df([[0]],"a:int").transform(a_transformer)
            df2 = dag.df([[0]],"b:int")
            dfs1 = WorkflowDataFrames(df1, df2)  # as array
            dfs2 = WorkflowDataFrames([df1, df2])  # as array
            dfs3 = WorkflowDataFrames(a=df1, b=df2)  # as dict
            dfs4 = WorkflowDataFrames(dict(a=df1, b=df2))  # as dict
            dfs5 = WorkflowDataFrames(dfs4, c=df2)  # copy and update
            dfs5["b"].show()  # how you get element when it's a dict
            dfs1[0].show()  # how you get element when it's an array
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._parent: Optional['FugueWorkflow'] = None
        super().__init__(*args, **kwargs)

    @property
    def workflow(self) -> 'FugueWorkflow':
        """The parent workflow"""
        assert_or_throw(self._parent is not None, ValueError('parent workflow is unknown'))
        return self._parent

    def __setitem__(self, key: str, value: WorkflowDataFrame, *args: Any, **kwds: Any) -> None:
        assert_or_throw(isinstance(value, WorkflowDataFrame), lambda: ValueError(f'{key}:{value} is not WorkflowDataFrame)'))
        if self._parent is None:
            self._parent = value.workflow
        else:
            assert_or_throw(self._parent is value.workflow, ValueError('different parent workflow detected in dataframes'))
        super().__setitem__(key, value, *args, **kwds)

    def __getitem__(self, key: Union[str, int]) -> WorkflowDataFrame:
        return super().__getitem__(key)

    def __getattr__(self, name: str) -> Any:
        """The dummy method to avoid PyLint complaint"""
        raise AttributeError(name)