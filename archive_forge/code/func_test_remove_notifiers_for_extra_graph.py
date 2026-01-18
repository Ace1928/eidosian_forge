import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_types import Instance, Int
from traits.observation.api import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation.expression import compile_expr, trait
from traits.observation.observe import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_remove_notifiers_for_extra_graph(self):
    observable = DummyObservable()
    extra_notifier = DummyNotifier()
    extra_observer = DummyObserver(observables=[observable], notifier=extra_notifier)
    extra_graph = ObserverGraph(node=extra_observer)
    observer = DummyObserver(extra_graphs=[extra_graph])
    graph = ObserverGraph(node=observer)
    observable.notifiers = [extra_notifier]
    call_add_or_remove_notifiers(graph=graph, remove=True)
    self.assertEqual(observable.notifiers, [])