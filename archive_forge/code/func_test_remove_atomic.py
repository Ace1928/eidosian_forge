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
def test_remove_atomic(self):
    notifier = DummyNotifier()
    maintainer = DummyNotifier()
    observable1 = DummyObservable()
    observable1.notifiers = [notifier, maintainer]
    old_observable1_notifiers = observable1.notifiers.copy()
    observable2 = DummyObservable()
    observable2.notifiers = [maintainer]
    old_observable2_notifiers = observable2.notifiers.copy()
    observable3 = DummyObservable()
    observable3.notifiers = [notifier, maintainer]
    old_observable3_notifiers = observable3.notifiers.copy()
    observer = DummyObserver(notify=True, observables=[observable1, observable2, observable3], notifier=notifier, maintainer=maintainer)
    graph = create_graph(observer, DummyObserver())
    with self.assertRaises(NotifierNotFound):
        call_add_or_remove_notifiers(object=mock.Mock(), graph=graph, remove=True)
    self.assertCountEqual(observable1.notifiers, old_observable1_notifiers)
    self.assertCountEqual(observable2.notifiers, old_observable2_notifiers)
    self.assertCountEqual(observable3.notifiers, old_observable3_notifiers)