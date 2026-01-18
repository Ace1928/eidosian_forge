import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_notify_reset(self):

    def call_me(state, details):
        pass
    notifier = nt.Notifier()
    notifier.register(nt.Notifier.ANY, call_me)
    self.assertEqual(1, len(notifier))
    notifier.reset()
    self.assertEqual(0, len(notifier))