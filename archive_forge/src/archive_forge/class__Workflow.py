import concurrent.futures as cf
import logging
import sys
from abc import ABC, abstractmethod
from enum import Enum
from threading import Event, RLock
from traceback import StackSummary, extract_stack
from typing import (
from uuid import uuid4
from adagio.exceptions import AbortedError, SkippedError, WorkflowBug
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from six import reraise  # type: ignore
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.assertion import assert_or_throw as aot
from triad.utils.convert import to_instance
from triad.utils.hash import to_uuid
class _Workflow(_Task):

    def __init__(self, spec: WorkflowSpec, ctx: WorkflowContext, parent_workflow: Optional['_Workflow']=None):
        super().__init__(spec, ctx, parent_workflow)
        self.tasks = IndexedOrderedDict()

    def _init_tasks(self):
        for k, v in self.spec.tasks.items():
            self.tasks[k] = self._build_task(v)
        self._set_outputs()

    def _build_task(self, spec: TaskSpec) -> _Task:
        if isinstance(spec, WorkflowSpec):
            task: _Task = _Workflow(spec, self.ctx, self)
        else:
            task = _Task(spec, self.ctx, self)
        self._set_configs(task, spec)
        self._set_inputs(task, spec)
        if isinstance(task, _Workflow):
            task._init_tasks()
        return task

    def _set_inputs(self, task: _Task, spec: TaskSpec) -> None:
        for f, to_expr in spec.node_spec.dependency.items():
            t = to_expr.split('.', 1)
            if len(t) == 1:
                task.inputs[f].set_dependency(self.inputs[t[0]])
            else:
                task.inputs[f].set_dependency(self.tasks[t[0]].outputs[t[1]])

    def _set_configs(self, task: _Task, spec: TaskSpec) -> None:
        for f, v in spec.node_spec.config.items():
            task.configs[f].set(v)
        for f, t in spec.node_spec.config_dependency.items():
            task.configs[f].set_dependency(self.configs[t])

    def _set_outputs(self) -> None:
        assert isinstance(self.spec, WorkflowSpec)
        for f, to_expr in self.spec.internal_dependency.items():
            t = to_expr.split('.', 1)
            if len(t) == 1:
                self.outputs[f].set_dependency(self.inputs[t[0]])
            else:
                self.outputs[f].set_dependency(self.tasks[t[0]].outputs[t[1]])

    def _register(self, temp: List[_Task]) -> None:
        for n in self.tasks.values():
            n._register(temp)

    def update_by_cache(self) -> None:
        self._ensure_fully_connected()
        for n in self.tasks.values():
            n.task.update_by_cache()