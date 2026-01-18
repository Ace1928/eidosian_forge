import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_optional_trait_added(self):
    graph = create_graph(create_observer(name='value', notify=True, optional=True))
    handler = mock.Mock()
    not_an_has_traits_instance = mock.Mock()
    try:
        call_add_or_remove_notifiers(object=not_an_has_traits_instance, graph=graph, handler=handler)
    except Exception:
        self.fail('Optional flag should have been propagated.')