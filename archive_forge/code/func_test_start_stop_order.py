import collections
import functools
import threading
import time
from taskflow import test
from taskflow.utils import threading_utils as tu
def test_start_stop_order(self):
    start_events = collections.deque()
    death_events = collections.deque()

    def before_start(i, t):
        start_events.append((i, 'bs'))

    def before_join(i, t):
        death_events.append((i, 'bj'))
        self.death.set()

    def after_start(i, t):
        start_events.append((i, 'as'))

    def after_join(i, t):
        death_events.append((i, 'aj'))
    for i in range(0, self.thread_count):
        self.bundle.bind(lambda: tu.daemon_thread(_spinner, self.death), before_join=functools.partial(before_join, i), after_join=functools.partial(after_join, i), before_start=functools.partial(before_start, i), after_start=functools.partial(after_start, i))
    self.assertEqual(self.thread_count, self.bundle.start())
    self.assertEqual(self.thread_count, len(self.bundle))
    self.assertEqual(self.thread_count, self.bundle.stop())
    self.assertEqual(0, self.bundle.stop())
    self.assertTrue(self.death.is_set())
    expected_start_events = []
    for i in range(0, self.thread_count):
        expected_start_events.extend([(i, 'bs'), (i, 'as')])
    self.assertEqual(expected_start_events, list(start_events))
    expected_death_events = []
    j = self.thread_count - 1
    for _i in range(0, self.thread_count):
        expected_death_events.extend([(j, 'bj'), (j, 'aj')])
        j -= 1
    self.assertEqual(expected_death_events, list(death_events))