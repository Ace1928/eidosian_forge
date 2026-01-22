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
def _ensure_fully_connected(self) -> None:
    """By design, this should be called always when fully connected,
        but if this failed, it means there is a bug in the framework itself.
        """
    for k, v in self.configs.items():
        try:
            v.get()
        except Exception:
            raise WorkflowBug(f"BUG: config {k}'s value or dependency is not set")
    for k, vv in self.inputs.items():
        aot(vv.dependency is not None, lambda: WorkflowBug(f"BUG: input {k}'s dependency is not set"))