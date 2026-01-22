from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
class CreateConstColumn(Create):

    def __init__(self, value: Any, name: str=''):
        super().__init__('to_col', value, name)
        self._value = value
        self._name = name

    def create(self, op: QPDEngine, ctx: TaskContext) -> Any:
        return op(self._op_name, self._value, self._name)