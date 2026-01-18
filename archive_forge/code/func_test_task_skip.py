import threading
import time
from collections import OrderedDict
from threading import RLock
from time import sleep
from typing import Any, Tuple
from adagio.exceptions import AbortedError
from adagio.instances import (NoOpCache, ParallelExecutionEngine, TaskContext,
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import InputSpec, OutputSpec, WorkflowSpec, _NodeSpec
from pytest import raises
from triad.exceptions import InvalidOperationError
from timeit import timeit
def test_task_skip():
    ts = build_task(example_helper1, t1, inputs=dict(a=1), configs=dict(b='xx'))
    assert _State.CREATED == ts.state
    ts.skip()
    assert _State.SKIPPED == ts.state
    raises(InvalidOperationError, lambda: ts.skip())
    ts.run()
    assert _State.SKIPPED == ts.state