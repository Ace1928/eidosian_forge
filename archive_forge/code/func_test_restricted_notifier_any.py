import collections
import functools
from taskflow import states
from taskflow import test
from taskflow.types import notifier as nt
def test_restricted_notifier_any(self):
    notifier = nt.RestrictedNotifier(['a', 'b'])
    self.assertRaises(ValueError, notifier.register, 'c', lambda *args, **kargs: None)
    notifier.register('b', lambda *args, **kargs: None)
    self.assertEqual(1, len(notifier))
    notifier.register(nt.RestrictedNotifier.ANY, lambda *args, **kargs: None)
    self.assertEqual(2, len(notifier))