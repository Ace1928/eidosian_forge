import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_selective_notify(self):
    call_counts = collections.defaultdict(list)

    def call_me_on(registered_state, state, details):
        call_counts[registered_state].append((state, details))
    notifier = nt.Notifier()
    call_me_on_success = functools.partial(call_me_on, states.SUCCESS)
    notifier.register(states.SUCCESS, call_me_on_success)
    self.assertTrue(notifier.is_registered(states.SUCCESS, call_me_on_success))
    call_me_on_any = functools.partial(call_me_on, nt.Notifier.ANY)
    notifier.register(nt.Notifier.ANY, call_me_on_any)
    self.assertTrue(notifier.is_registered(nt.Notifier.ANY, call_me_on_any))
    self.assertEqual(2, len(notifier))
    notifier.notify(states.SUCCESS, {})
    self.assertEqual(1, len(call_counts[nt.Notifier.ANY]))
    self.assertEqual(1, len(call_counts[states.SUCCESS]))
    notifier.notify(states.FAILURE, {})
    self.assertEqual(2, len(call_counts[nt.Notifier.ANY]))
    self.assertEqual(1, len(call_counts[states.SUCCESS]))
    self.assertEqual(2, len(call_counts))