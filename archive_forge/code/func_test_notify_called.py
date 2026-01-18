import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_notify_called(self):
    call_collector = []

    def call_me(state, details):
        call_collector.append((state, details))
    notifier = nt.Notifier()
    notifier.register(nt.Notifier.ANY, call_me)
    notifier.notify(states.SUCCESS, {})
    notifier.notify(states.SUCCESS, {})
    self.assertEqual(2, len(call_collector))
    self.assertEqual(1, len(notifier))