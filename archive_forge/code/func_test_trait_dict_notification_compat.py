import unittest
from traits.observation._dict_change_event import (
from traits.trait_dict_object import TraitDict
def test_trait_dict_notification_compat(self):
    events = []

    def notifier(*args, **kwargs):
        event = dict_event_factory(*args, **kwargs)
        events.append(event)
    trait_dict = TraitDict({'3': 3, '4': 4}, notifiers=[notifier])
    del trait_dict['4']
    event, = events
    self.assertIs(event.object, trait_dict)
    self.assertEqual(event.removed, {'4': 4})
    events.clear()
    trait_dict.update({'3': None, '1': 1})
    event, = events
    self.assertEqual(event.removed, {'3': 3})
    self.assertEqual(event.added, {'3': None, '1': 1})