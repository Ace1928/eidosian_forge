from typing import Any
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.utils.io import exists, join, makedirs, rm
from fugue.collections.partition import PartitionSpec
from fugue.collections.yielded import PhysicalYielded
from fugue.constants import FUGUE_CONF_WORKFLOW_CHECKPOINT_PATH
from fugue.dataframe import DataFrame
from fugue.exceptions import FugueWorkflowCompileError, FugueWorkflowRuntimeError
from fugue.execution.execution_engine import ExecutionEngine
class WeakCheckpoint(Checkpoint):

    def __init__(self, lazy: bool=False, **kwargs: Any):
        super().__init__(deterministic=False, lazy=lazy, **kwargs)

    def run(self, df: DataFrame, path: 'CheckpointPath') -> DataFrame:
        return path.execution_engine.persist(df, lazy=self.lazy, **self.kwargs)

    @property
    def is_null(self) -> bool:
        return False