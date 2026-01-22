import contextlib
import itertools
from unittest import mock
import eventlet
from heat.common import exception
from heat.common import timeutils
from heat.engine import dependencies
from heat.engine import scheduler
from heat.tests import common
class DependencyTaskGroupTest(common.HeatTestCase):

    def setUp(self):
        super(DependencyTaskGroupTest, self).setUp()
        self.aggregate_exceptions = False
        self.error_wait_time = None
        self.reverse_order = False

    @contextlib.contextmanager
    def _dep_test(self, *edges):
        dummy = DummyTask(getattr(self, 'steps', 3))
        deps = dependencies.Dependencies(edges)
        tg = scheduler.DependencyTaskGroup(deps, dummy, reverse=self.reverse_order, error_wait_time=self.error_wait_time, aggregate_exceptions=self.aggregate_exceptions)
        tracker = StepTracker()
        yield tracker
        dummy.do_step = mock.Mock(side_effect=tracker.side_effect)
        scheduler.TaskRunner(tg)(wait_time=None)
        tracker.verify_calls(dummy.do_step)

    def test_test(self):

        def failing_test():
            with self._dep_test(('only', None)) as track:
                track.expect_call(1, 'only')
                track.expect_call(3, 'only')
        self.assertRaises(AssertionError, failing_test)

    @mock.patch.object(scheduler.TaskRunner, '_sleep')
    def test_no_steps(self, mock_sleep):
        self.steps = 0
        with self._dep_test(('second', 'first')):
            pass

    def test_single_node(self):
        with self._dep_test(('only', None)) as track:
            track.expect_call(1, 'only')
            track.expect_call(2, 'only')
            track.expect_call(3, 'only')

    def test_disjoint(self):
        with self._dep_test(('1', None), ('2', None)) as track:
            track.expect_call_group(1, ('1', '2'))
            track.expect_call_group(2, ('1', '2'))
            track.expect_call_group(3, ('1', '2'))

    def test_single_fwd(self):
        with self._dep_test(('second', 'first')) as track:
            track.expect_call(1, 'first')
            track.expect_call(2, 'first')
            track.expect_call(3, 'first')
            track.expect_call(1, 'second')
            track.expect_call(2, 'second')
            track.expect_call(3, 'second')

    def test_chain_fwd(self):
        with self._dep_test(('third', 'second'), ('second', 'first')) as track:
            track.expect_call(1, 'first')
            track.expect_call(2, 'first')
            track.expect_call(3, 'first')
            track.expect_call(1, 'second')
            track.expect_call(2, 'second')
            track.expect_call(3, 'second')
            track.expect_call(1, 'third')
            track.expect_call(2, 'third')
            track.expect_call(3, 'third')

    def test_diamond_fwd(self):
        with self._dep_test(('last', 'mid1'), ('last', 'mid2'), ('mid1', 'first'), ('mid2', 'first')) as track:
            track.expect_call(1, 'first')
            track.expect_call(2, 'first')
            track.expect_call(3, 'first')
            track.expect_call_group(1, ('mid1', 'mid2'))
            track.expect_call_group(2, ('mid1', 'mid2'))
            track.expect_call_group(3, ('mid1', 'mid2'))
            track.expect_call(1, 'last')
            track.expect_call(2, 'last')
            track.expect_call(3, 'last')

    def test_complex_fwd(self):
        with self._dep_test(('last', 'mid1'), ('last', 'mid2'), ('mid1', 'mid3'), ('mid1', 'first'), ('mid3', 'first'), ('mid2', 'first')) as track:
            track.expect_call(1, 'first')
            track.expect_call(2, 'first')
            track.expect_call(3, 'first')
            track.expect_call_group(1, ('mid2', 'mid3'))
            track.expect_call_group(2, ('mid2', 'mid3'))
            track.expect_call_group(3, ('mid2', 'mid3'))
            track.expect_call(1, 'mid1')
            track.expect_call(2, 'mid1')
            track.expect_call(3, 'mid1')
            track.expect_call(1, 'last')
            track.expect_call(2, 'last')
            track.expect_call(3, 'last')

    def test_many_edges_fwd(self):
        with self._dep_test(('last', 'e1'), ('last', 'mid1'), ('last', 'mid2'), ('mid1', 'e2'), ('mid1', 'mid3'), ('mid2', 'mid3'), ('mid3', 'e3')) as track:
            track.expect_call_group(1, ('e1', 'e2', 'e3'))
            track.expect_call_group(2, ('e1', 'e2', 'e3'))
            track.expect_call_group(3, ('e1', 'e2', 'e3'))
            track.expect_call(1, 'mid3')
            track.expect_call(2, 'mid3')
            track.expect_call(3, 'mid3')
            track.expect_call_group(1, ('mid2', 'mid1'))
            track.expect_call_group(2, ('mid2', 'mid1'))
            track.expect_call_group(3, ('mid2', 'mid1'))
            track.expect_call(1, 'last')
            track.expect_call(2, 'last')
            track.expect_call(3, 'last')

    def test_dbldiamond_fwd(self):
        with self._dep_test(('last', 'a1'), ('last', 'a2'), ('a1', 'b1'), ('a2', 'b1'), ('a2', 'b2'), ('b1', 'first'), ('b2', 'first')) as track:
            track.expect_call(1, 'first')
            track.expect_call(2, 'first')
            track.expect_call(3, 'first')
            track.expect_call_group(1, ('b1', 'b2'))
            track.expect_call_group(2, ('b1', 'b2'))
            track.expect_call_group(3, ('b1', 'b2'))
            track.expect_call_group(1, ('a1', 'a2'))
            track.expect_call_group(2, ('a1', 'a2'))
            track.expect_call_group(3, ('a1', 'a2'))
            track.expect_call(1, 'last')
            track.expect_call(2, 'last')
            track.expect_call(3, 'last')

    def test_circular_deps(self):
        d = dependencies.Dependencies([('first', 'second'), ('second', 'third'), ('third', 'first')])
        self.assertRaises(exception.CircularDependencyException, scheduler.DependencyTaskGroup, d)

    def test_aggregate_exceptions_raises_all_at_the_end(self):

        def run_tasks_with_exceptions(e1=None, e2=None):
            self.aggregate_exceptions = True
            tasks = (('A', None), ('B', None), ('C', None))
            with self._dep_test(*tasks) as track:
                track.expect_call_group(1, ('A', 'B', 'C'))
                track.raise_on(1, 'C', e1)
                track.expect_call_group(2, ('A', 'B'))
                track.raise_on(2, 'B', e2)
                track.expect_call(3, 'A')
        e1 = Exception('e1')
        e2 = Exception('e2')
        exc = self.assertRaises(scheduler.ExceptionGroup, run_tasks_with_exceptions, e1, e2)
        self.assertEqual(set([e1, e2]), set(exc.exceptions))

    def test_aggregate_exceptions_cancels_dependent_tasks_recursively(self):

        def run_tasks_with_exceptions(e1=None, e2=None):
            self.aggregate_exceptions = True
            tasks = (('A', None), ('B', 'A'), ('C', 'B'))
            with self._dep_test(*tasks) as track:
                track.expect_call(1, 'A')
                track.raise_on(1, 'A', e1)
        e1 = Exception('e1')
        exc = self.assertRaises(scheduler.ExceptionGroup, run_tasks_with_exceptions, e1)
        self.assertEqual([e1], exc.exceptions)

    def test_aggregate_exceptions_cancels_tasks_in_reverse_order(self):

        def run_tasks_with_exceptions(e1=None, e2=None):
            self.reverse_order = True
            self.aggregate_exceptions = True
            tasks = (('A', None), ('B', 'A'), ('C', 'B'))
            with self._dep_test(*tasks) as track:
                track.expect_call(1, 'C')
                track.raise_on(1, 'C', e1)
        e1 = Exception('e1')
        exc = self.assertRaises(scheduler.ExceptionGroup, run_tasks_with_exceptions, e1)
        self.assertEqual([e1], exc.exceptions)

    def test_exceptions_on_cancel(self):

        class TestException(Exception):
            pass

        class ExceptionOnExit(Exception):
            pass
        cancelled = []

        def task_func(arg):
            for i in range(4):
                if i > 1:
                    raise TestException
                try:
                    yield
                except GeneratorExit:
                    cancelled.append(arg)
                    raise ExceptionOnExit
        tasks = (('A', None), ('B', None), ('C', None))
        deps = dependencies.Dependencies(tasks)
        tg = scheduler.DependencyTaskGroup(deps, task_func)
        task = tg()
        next(task)
        next(task)
        self.assertRaises(TestException, next, task)
        self.assertEqual(len(tasks) - 1, len(cancelled))

    def test_exception_grace_period(self):
        e1 = Exception('e1')

        def run_tasks_with_exceptions():
            self.error_wait_time = 5
            tasks = (('A', None), ('B', None), ('C', 'A'))
            with self._dep_test(*tasks) as track:
                track.expect_call_group(1, ('A', 'B'))
                track.expect_call_group(2, ('A', 'B'))
                track.raise_on(2, 'B', e1)
                track.expect_call(3, 'B')
        exc = self.assertRaises(type(e1), run_tasks_with_exceptions)
        self.assertEqual(e1, exc)

    def test_exception_grace_period_expired(self):
        e1 = Exception('e1')

        def run_tasks_with_exceptions():
            self.steps = 5
            self.error_wait_time = 0.05

            def sleep():
                eventlet.sleep(self.error_wait_time)
            tasks = (('A', None), ('B', None), ('C', 'A'))
            with self._dep_test(*tasks) as track:
                track.expect_call_group(1, ('A', 'B'))
                track.expect_call_group(2, ('A', 'B'))
                track.raise_on(2, 'B', e1)
                track.expect_call(3, 'B')
                track.expect_call(4, 'B')
                track.sleep_on(4, 'B', self.error_wait_time)
        exc = self.assertRaises(type(e1), run_tasks_with_exceptions)
        self.assertEqual(e1, exc)

    def test_exception_grace_period_per_task(self):
        e1 = Exception('e1')

        def get_wait_time(key):
            if key == 'B':
                return 5
            else:
                return None

        def run_tasks_with_exceptions():
            self.error_wait_time = get_wait_time
            tasks = (('A', None), ('B', None), ('C', 'A'))
            with self._dep_test(*tasks) as track:
                track.expect_call_group(1, ('A', 'B'))
                track.expect_call_group(2, ('A', 'B'))
                track.raise_on(2, 'A', e1)
                track.expect_call(3, 'B')
        exc = self.assertRaises(type(e1), run_tasks_with_exceptions)
        self.assertEqual(e1, exc)

    def test_thrown_exception_order(self):
        e1 = Exception('e1')
        e2 = Exception('e2')
        tasks = (('A', None), ('B', None), ('C', 'A'))
        deps = dependencies.Dependencies(tasks)
        tg = scheduler.DependencyTaskGroup(deps, DummyTask(), reverse=self.reverse_order, error_wait_time=1, aggregate_exceptions=self.aggregate_exceptions)
        task = tg()
        next(task)
        task.throw(e1)
        next(task)
        tg.error_wait_time = None
        exc = self.assertRaises(type(e2), task.throw, e2)
        self.assertIs(e2, exc)