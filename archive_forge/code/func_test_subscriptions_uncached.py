import unittest
from zope.interface.tests import OptimizationTestMixin
def test_subscriptions_uncached(self):
    _called_with = []
    _results = [object(), object(), object()]

    def _subscriptions(self, required, provided):
        _called_with.append((required, provided))
        return tuple(_results)
    lb = self._makeOne(uc_subscriptions=_subscriptions)
    found = lb.subscriptions('A', 'B')
    self.assertEqual(found, tuple(_results))
    self.assertEqual(_called_with, [(('A',), 'B')])