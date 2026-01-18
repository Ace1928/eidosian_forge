import unittest
from traits.api import (
from traits.observation.api import (
def test_observe_instance_in_set(self):
    container = ClassWithSetOfInstance()
    events = []
    handler = events.append
    container.observe(handler=handler, expression=trait('instances', notify=False).set_items(notify=False).trait('value'))
    single_value_instance = SingleValue()
    container.instances = set([single_value_instance])
    self.assertEqual(len(events), 0)
    single_value_instance.value += 1
    event, = events
    self.assertEqual(event.object, single_value_instance)
    self.assertEqual(event.name, 'value')
    self.assertEqual(event.old, 0)
    self.assertEqual(event.new, 1)