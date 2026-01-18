from typing import Any, Dict, Union
from fugue.collections.partition import PartitionCursor, PartitionSpec
from fugue.dataframe import DataFrame, DataFrames
from fugue.execution.execution_engine import ExecutionEngine
from fugue.extensions._utils import validate_input_schema, validate_partition_spec
from fugue.rpc import RPCClient, RPCServer
from triad.collections import ParamDict, Schema
from triad.utils.convert import get_full_type_path
from triad.utils.hash import to_uuid
def validate_on_runtime(self, data: Union[DataFrame, DataFrames]) -> None:
    if isinstance(data, DataFrame):
        validate_input_schema(data.schema, self.validation_rules)
    else:
        for df in data.values():
            validate_input_schema(df.schema, self.validation_rules)