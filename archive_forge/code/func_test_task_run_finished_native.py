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
def test_task_run_finished_native():
    ts = build_task(example_helper1, None, inputs=dict(a=1), configs=dict(b='xx'))
    assert _State.CREATED == ts.state
    ts.run()
    assert _State.FINISHED == ts.state
    assert ts.outputs['_0'].is_successful
    assert 3 == ts.outputs['_0'].value
    assert ts.outputs['_1'].is_successful
    assert -1 == ts.outputs['_1'].value