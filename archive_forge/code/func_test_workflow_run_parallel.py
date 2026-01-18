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
def test_workflow_run_parallel():
    s = SimpleSpec()
    s.add('a', wait_task0)
    s.add('b', wait_task0)
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(2, ctx)
    ctx.run(s, {})
    expected = {'a': 1, 'b': 1}
    for k, v in expected.items():
        assert v == hooks.res[k]
    s = SimpleSpec()
    s.add('a', wait_task0)
    s.add('b', wait_task0e)
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(2, ctx)
    with raises(NotImplementedError):
        ctx.run(s, {})
    expected = {'a': 1}
    for k, v in expected.items():
        assert v == hooks.res[k]
    s = SimpleSpec()
    s.add('a', wait_task0)
    s.add('b', wait_task0e)
    s.add('c', wait_task1, 'a')
    s.add('d', wait_task1, 'b')
    s.add('e', wait_task1, 'd')
    s.add('f', wait_task1, 'c')
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(10, ctx)

    def run():
        with raises(NotImplementedError):
            ctx.run(s, {})
    t = timeit(run, number=1)
    assert t < 0.2
    expected = {'a': 1}
    for k, v in expected.items():
        assert v == hooks.res[k]
    assert 'b' in hooks.failed
    s = SimpleSpec()
    s.add('a', wait_task0)
    s.add('b', wait_task0)
    s.add('c', wait_task1, 'a')
    s.add('d', wait_task1, 'b')
    s.add('e', wait_task1, 'd')
    s.add('f', wait_task1, 'c')
    hooks = MockHooks(None)
    ctx = WorkflowContext(hooks=hooks)
    ctx._engine = ParallelExecutionEngine(2, ctx)
    t = timeit(lambda: ctx.run(s, {}), number=1)
    assert t < 0.4
    assert 3 == hooks.res['e']
    assert 3 == hooks.res['f']