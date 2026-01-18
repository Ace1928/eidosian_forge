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
def test_task_run_cached_non_deterministic():
    cache = MockCache()
    ts = build_task(example_helper1, t2, inputs=dict(a=1), configs=dict(b='xx'), cache=cache, deterministic=False)
    id1 = ts.__uuid__()
    oid1 = ts.outputs['_0'].__uuid__()
    assert 0 == cache.get_called
    assert 0 == cache.set_called
    assert 0 == cache.skip_called
    assert _State.CREATED == ts.state
    ts.run()
    assert _State.FINISHED == ts.state
    assert ts.outputs['_0'].is_successful
    assert 3 == ts.outputs['_0'].value
    assert ts.outputs['_1'].is_successful
    assert -1 == ts.outputs['_1'].value
    assert 0 == cache.get_called
    assert 0 == cache.set_called
    assert 0 == cache.skip_called
    assert id1 == ts.__uuid__()
    assert oid1 == ts.outputs['_0'].__uuid__()
    assert 0 == len(cache.tb)
    ts = build_task(example_helper1, t2, inputs=dict(a=1), configs=dict(b='xx'), cache=cache, deterministic=False)
    assert id1 != ts.__uuid__()
    assert oid1 != ts.outputs['_0'].__uuid__()
    assert 0 == cache.get_called
    assert 0 == cache.set_called
    assert 0 == cache.skip_called
    assert _State.CREATED == ts.state