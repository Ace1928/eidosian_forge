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
class WorkflowExecutionEngine(WorkflowContextMember, ABC):

    def __init__(self, wf_ctx: 'WorkflowContext'):
        super().__init__(wf_ctx)

    def run(self, spec: WorkflowSpec, configs: Dict[str, Any]) -> None:
        wf = _make_top_level_workflow(spec, self.context, configs)
        tasks_to_run = self.preprocess(wf)
        self.run_tasks(tasks_to_run)

    @abstractmethod
    def preprocess(self, wf: '_Workflow') -> List['_Task']:
        raise NotImplementedError

    @abstractmethod
    def run_tasks(self, tasks: List['_Task']) -> None:
        raise NotImplementedError