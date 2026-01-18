import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
def test_notify_filter_values_changed(self):
    instance = DummyParent()
    observer = create_observer(filter=lambda name, trait: type(trait.trait_type) is Int)
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance, graph=create_graph(observer), handler=handler)
    instance.number += 1
    self.assertEqual(handler.call_count, 1)
    handler.reset_mock()
    instance.number2 += 1
    self.assertEqual(handler.call_count, 1)