import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
def test_trait_added_filtered_matched(self):
    instance = DummyParent()
    integer_observer = create_observer(filter=lambda name, trait: type(trait.trait_type) is Int, notify=True)
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance, graph=create_graph(integer_observer), handler=handler)
    instance.add_trait('another_number', Int())
    instance.another_number += 1
    self.assertEqual(handler.call_count, 1)