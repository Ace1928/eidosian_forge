import unittest
from traits.api import (
from traits.observation.api import (
def test_observe_decorator_anytrait(self):
    events = []
    obj = HasVariousTraits(trait_change_callback=events.append)
    obj.foo = 23
    obj.bar = 'on'
    self.assertEqual(len(events), 3)
    callback_event, foo_event, bar_event = events
    self.assertEqual(callback_event.object, obj)
    self.assertEqual(callback_event.name, 'trait_change_callback')
    self.assertIs(callback_event.old, None)
    self.assertEqual(callback_event.new, events.append)
    self.assertEqual(foo_event.object, obj)
    self.assertEqual(foo_event.name, 'foo')
    self.assertEqual(foo_event.old, 16)
    self.assertEqual(foo_event.new, 23)
    self.assertEqual(bar_event.object, obj)
    self.assertEqual(bar_event.name, 'bar')
    self.assertEqual(bar_event.old, 'off')
    self.assertEqual(bar_event.new, 'on')