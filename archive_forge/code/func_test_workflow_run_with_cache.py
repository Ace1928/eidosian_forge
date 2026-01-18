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
def test_workflow_run_with_cache():
    s = SimpleSpec()
    s.add('a', example_helper_task0)
    s.add('c', example_helper_task1)
    cache = MockCache()
    hooks1 = MockHooks(None)
    ctx = WorkflowContext(cache=cache, hooks=hooks1)
    ctx.run(s, {})
    assert 2 == cache.get_called
    assert 0 == cache.hit
    assert 2 == cache.set_called
    expected = {'a': 10, 'c': 11}
    for k, v in expected.items():
        assert v == hooks1.res[k]
    ctx.run(s, {})
    assert 4 == cache.get_called
    assert 2 == cache.hit
    assert 2 == cache.set_called
    expected = {'a': 10, 'c': 11}
    for k, v in expected.items():
        assert v == hooks1.res[k]
    hooks2 = MockHooks(None)
    ctx = WorkflowContext(cache=cache, hooks=hooks2)
    ctx.run(s, {})
    assert 6 == cache.get_called
    assert 4 == cache.hit
    assert 2 == cache.set_called
    expected = {'a': 10, 'c': 11}
    for k, v in expected.items():
        assert v == hooks1.res[k]