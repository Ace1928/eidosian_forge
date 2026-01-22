from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
class QPDWorkflowContext(WorkflowContext):

    def __init__(self, qpd_engine: QPDEngine, dfs: Dict[str, Any]):
        self._qpd_engine = qpd_engine
        self._dfs = DataFrames({k: qpd_engine.to_df(v) for k, v in dfs.items()})
        self._result: Any = None
        super().__init__()

    @property
    def qpd_engine(self) -> QPDEngine:
        return self._qpd_engine

    @property
    def dfs(self) -> DataFrames:
        return self._dfs

    def set_result(self, obj: Any) -> None:
        self._result = obj

    @property
    def result(self) -> Any:
        return self._result