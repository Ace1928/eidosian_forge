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
class TestObserveAddNotifier(unittest.TestCase):
    """ Test the add_notifiers action."""

    def test_add_trait_notifiers(self):
        observable = DummyObservable()
        notifier = DummyNotifier()
        observer = DummyObserver(notify=True, observables=[observable], notifier=notifier)
        graph = ObserverGraph(node=observer)
        call_add_or_remove_notifiers(graph=graph, remove=False)
        self.assertEqual(observable.notifiers, [notifier])

    def test_add_trait_notifiers_notify_flag_is_false(self):
        observable = DummyObservable()
        notifier = DummyNotifier()
        observer = DummyObserver(notify=False, observables=[observable], notifier=notifier)
        graph = ObserverGraph(node=observer)
        call_add_or_remove_notifiers(graph=graph, remove=False)
        self.assertEqual(observable.notifiers, [])

    def test_add_maintainers(self):
        observable = DummyObservable()
        maintainer = DummyNotifier()
        root_observer = DummyObserver(notify=False, observables=[observable], maintainer=maintainer)
        graph = ObserverGraph(node=root_observer, children=[ObserverGraph(node=DummyObserver()), ObserverGraph(node=DummyObserver())])
        call_add_or_remove_notifiers(graph=graph, remove=False)
        self.assertEqual(observable.notifiers, [maintainer, maintainer])

    def test_add_notifiers_for_children_graphs(self):
        observable1 = DummyObservable()
        child_observer1 = DummyObserver(observables=[observable1])
        observable2 = DummyObservable()
        child_observer2 = DummyObserver(observables=[observable2])
        parent_observer = DummyObserver(next_objects=[mock.Mock()])
        graph = ObserverGraph(node=parent_observer, children=[ObserverGraph(node=child_observer1), ObserverGraph(node=child_observer2)])
        call_add_or_remove_notifiers(graph=graph, remove=False)
        self.assertCountEqual(observable1.notifiers, [child_observer1.notifier])
        self.assertCountEqual(observable2.notifiers, [child_observer2.notifier])

    def test_add_notifiers_for_extra_graph(self):
        observable = DummyObservable()
        extra_notifier = DummyNotifier()
        extra_observer = DummyObserver(observables=[observable], notifier=extra_notifier)
        extra_graph = ObserverGraph(node=extra_observer)
        observer = DummyObserver(extra_graphs=[extra_graph])
        graph = ObserverGraph(node=observer)
        call_add_or_remove_notifiers(graph=graph, remove=False)
        self.assertEqual(observable.notifiers, [extra_notifier])

    def test_add_notifier_atomic(self):

        class BadNotifier(DummyNotifier):

            def add_to(self, observable):
                raise ZeroDivisionError()
        observable = DummyObservable()
        good_observer = DummyObserver(notify=True, observables=[observable], next_objects=[mock.Mock()], notifier=DummyNotifier(), maintainer=DummyNotifier())
        bad_observer = DummyObserver(notify=True, observables=[observable], notifier=BadNotifier(), maintainer=DummyNotifier())
        graph = create_graph(good_observer, bad_observer)
        with self.assertRaises(ZeroDivisionError):
            call_add_or_remove_notifiers(object=mock.Mock(), graph=graph)
        self.assertEqual(observable.notifiers, [])