import unittest
from traits.observation._list_change_event import (
from traits.trait_list_object import TraitList
class TestListEventFactory(unittest.TestCase):
    """ Test event factory compatibility with TraitList.notify """

    def test_trait_list_notification_compat(self):
        events = []

        def notifier(*args, **kwargs):
            event = list_event_factory(*args, **kwargs)
            events.append(event)
        trait_list = TraitList([0, 1, 2], notifiers=[notifier])
        trait_list[1:] = [3, 4]
        event, = events
        self.assertIsInstance(event, ListChangeEvent)
        self.assertIs(event.object, trait_list)
        self.assertEqual(event.index, 1)
        self.assertEqual(event.removed, [1, 2])
        self.assertEqual(event.added, [3, 4])