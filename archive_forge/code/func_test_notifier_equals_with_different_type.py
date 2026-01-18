import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_notifier_equals_with_different_type(self):
    notifier = create_notifier()
    self.assertFalse(notifier.equals(str))