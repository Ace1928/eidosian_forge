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
class WorkflowContextMember(object):
    """Base class for components of func:`~adagio.instances.WorkflowContext`

    :param wf_ctx: parent workflow context
    """

    def __init__(self, wf_ctx: 'WorkflowContext'):
        self._wf_ctx = wf_ctx

    @property
    def context(self) -> 'WorkflowContext':
        """parent workflow context"""
        return self._wf_ctx

    @property
    def conf(self) -> ParamDict:
        """config of parent workflow context"""
        return self.context.conf