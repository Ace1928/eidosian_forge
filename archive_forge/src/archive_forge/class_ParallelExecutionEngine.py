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
class ParallelExecutionEngine(WorkflowExecutionEngine):

    def __init__(self, concurrency: int, wf_ctx: 'WorkflowContext'):
        super().__init__(wf_ctx)
        self._concurrency = concurrency

    def preprocess(self, wf: '_Workflow') -> List['_Task']:
        temp: List['_Task'] = []
        wf._register(temp)
        if self._concurrency <= 1:
            return temp
        return [t for t in temp if len(t.upstream) == 0]

    def run_tasks(self, tasks: List['_Task']) -> None:
        if self._concurrency <= 1:
            for t in tasks:
                self.run_single(t)
            return
        with cf.ThreadPoolExecutor(max_workers=self._concurrency) as e:
            jobs = [e.submit(self.run_single, task) for task in tasks]
            while jobs:
                for f in cf.as_completed(jobs):
                    jobs.remove(f)
                    try:
                        for task in f.result().downstream:
                            jobs.append(e.submit(self.run_single, task))
                    except Exception:
                        self.context.abort()
                        raise

    def run_single(self, task: '_Task') -> '_Task':
        task.update_by_cache()
        task.run()
        task.reraise()
        return task