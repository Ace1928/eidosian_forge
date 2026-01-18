import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_remove_notifier(self):
    instance = DummyClass()
    notifier = create_notifier()
    notifier.add_to(instance)
    notifier.remove_from(instance)
    self.assertEqual(instance.notifiers, [])