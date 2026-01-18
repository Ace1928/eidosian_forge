from typing import Any, Dict, Union
from fugue.collections.partition import PartitionCursor, PartitionSpec
from fugue.dataframe import DataFrame, DataFrames
from fugue.execution.execution_engine import ExecutionEngine
from fugue.extensions._utils import validate_input_schema, validate_partition_spec
from fugue.rpc import RPCClient, RPCServer
from triad.collections import ParamDict, Schema
from triad.utils.convert import get_full_type_path
from triad.utils.hash import to_uuid
@property
def workflow_conf(self) -> ParamDict:
    """Workflow level configs, this is accessible even in
        :class:`~fugue.extensions.transformer.transformer.Transformer` and
        :class:`~fugue.extensions.transformer.transformer.CoTransformer`

        .. admonition:: Examples

            >>> dag = FugueWorkflow().df(...).transform(using=dummy)
            >>> dag.run(NativeExecutionEngine(conf={"b": 10}))

        You will get ``{"b": 10}`` as `workflow_conf` in the ``dummy`` transformer
        on both driver and workers.
        """
    if '_workflow_conf' in self.__dict__:
        return self._workflow_conf
    return self.execution_engine.conf