from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
class WorkflowColumn(Column, QPDTaskWrapper):

    def __init__(self, workflow: 'QPDWorkflow', task: QPDTask, col: str=''):
        Column.__init__(self, None, col)
        QPDTaskWrapper.__init__(self, workflow, task)

    @property
    def native(self) -> Any:
        raise NotImplementedError

    def rename(self, name) -> 'WorkflowColumn':
        if name == self.name:
            return self
        task = self.workflow.add('rename', self.task, name)
        return WorkflowColumn(self.workflow, task, name)