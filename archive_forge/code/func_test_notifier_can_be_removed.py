import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
def test_notifier_can_be_removed(self):

    def filter_func(name, trait):
        return name.startswith('num')
    instance = DummyParent()
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance, graph=create_graph(create_observer(filter=filter_func)), handler=handler)
    instance.number += 1
    self.assertEqual(handler.call_count, 1)
    handler.reset_mock()
    call_add_or_remove_notifiers(object=instance, graph=create_graph(create_observer(filter=filter_func)), handler=handler, remove=True)
    instance.number += 1
    self.assertEqual(handler.call_count, 0)