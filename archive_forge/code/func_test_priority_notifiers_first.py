import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def test_priority_notifiers_first(self):
    obj = DynamicNotifiers()
    expected_high = set([1, 3])
    expected_low = set([0, 2])
    obj.on_trait_change(obj.low_priority_first, 'priority_test')
    obj.on_trait_change(obj.high_priority_first, 'priority_test', priority=True)
    obj.on_trait_change(obj.low_priority_second, 'priority_test')
    obj.on_trait_change(obj.high_priority_second, 'priority_test', priority=True)
    obj.priority_test = None
    high = set(obj.prioritized_notifications[:2])
    low = set(obj.prioritized_notifications[2:])
    self.assertSetEqual(expected_high, high)
    self.assertSetEqual(expected_low, low)