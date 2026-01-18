import unittest
from traits.observation._set_change_event import (
from traits.trait_set_object import TraitSet
def test_trait_set_notification_compat(self):
    events = []

    def notifier(*args, **kwargs):
        event = set_event_factory(*args, **kwargs)
        events.append(event)
    trait_set = TraitSet([1, 2, 3], notifiers=[notifier])
    trait_set.add(4)
    event, = events
    self.assertIs(event.object, trait_set)
    self.assertEqual(event.added, {4})
    self.assertEqual(event.removed, set())
    events.clear()
    trait_set.remove(4)
    event, = events
    self.assertEqual(event.added, set())
    self.assertEqual(event.removed, {4})