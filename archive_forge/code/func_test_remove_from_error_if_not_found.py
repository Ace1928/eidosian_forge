import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_remove_from_error_if_not_found(self):
    dummy = DummyObservable()
    notifier1 = create_notifier()
    with self.assertRaises(NotifierNotFound) as e:
        notifier1.remove_from(dummy)
    self.assertEqual(str(e.exception), 'Notifier not found.')