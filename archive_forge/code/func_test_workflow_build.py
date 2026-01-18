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
def test_workflow_build():
    """
       a   d       j
       |   |      /        b   e     |   k
       |   |     |   |
       c   |     aa  bb <----- sub start
        \\ /      |   |           f       a   b   |
         |       |  /|   |
         g       | | c   |
        / \\      |  \\    |
       h   i     cc  dd  ee <----- sub end
    """
    s1 = SimpleSpec(['aa', 'bb'], ['cc', 'dd', 'ee'])
    s1.add('a', example_helper_task1, '*aa')
    s1.add('b', example_helper_task1, '*bb')
    s1.add('c', example_helper_task1)
    s1.link('cc', 'a._0')
    s1.link('dd', 'b._0')
    s1.link('ee', 'bb')
    s = SimpleSpec()
    s.add('a', example_helper_task0)
    s.add('b', example_helper_task1)
    s.add('c', example_helper_task1)
    s.add('d', example_helper_task0)
    s.add('e', example_helper_task1)
    s.add('f', example_helper_task2, 'c', 'e')
    s.add('g', example_helper_task3)
    s.add('h', example_helper_task3, 'g._0')
    s.add('i', example_helper_task3, 'g._1')
    s.add('j', example_helper_task0)
    s.add('k', example_helper_task1)
    s.add_task('x', s1, {'aa': 'j._0', 'bb': 'k._0'})
    ctx = WorkflowContext()
    raises(InvalidOperationError, lambda: _make_top_level_workflow(s1, ctx, {}))
    wf = _make_top_level_workflow(s, ctx, {})
    assert wf.tasks['a'].__uuid__() == wf.tasks['d'].__uuid__()
    assert wf.tasks['a'].execution_id != wf.tasks['d'].execution_id
    assert {wf.tasks['j']} == wf.tasks['x'].tasks['a'].upstream

    def assert_dep(node, up, down):
        assert set(list(up)) == (up if isinstance(up, set) else set((t.name for t in wf.tasks[node].upstream)))
        assert set(list(down)) == (down if isinstance(down, set) else set((t.name for t in wf.tasks[node].downstream)))
    assert_dep('a', '', 'b')
    assert_dep('b', 'a', 'c')
    assert_dep('c', 'b', 'f')
    assert_dep('d', '', 'e')
    assert_dep('e', 'd', 'f')
    assert_dep('f', 'ce', 'g')
    assert_dep('g', 'f', 'hi')
    assert_dep('h', 'g', '')
    assert_dep('i', 'g', '')
    assert_dep('j', '', 'ak')
    assert_dep('k', 'j', 'b')
    assert_dep('x', 'jk', '')