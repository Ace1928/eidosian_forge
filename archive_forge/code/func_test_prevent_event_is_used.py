import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_prevent_event_is_used(self):

    def prevent_event(event):
        return True
    handler = mock.Mock()
    notifier = create_notifier(handler=handler, prevent_event=prevent_event)
    notifier(a=1, b=2)
    handler.assert_not_called()