import unittest
from traits.api import (
from traits.observation.api import (
def test_observe_instance_in_nested_list(self):
    container = ClassWithListOfListOfInstance()
    events = []
    handler = events.append
    container.observe(expression=trait('list_of_list_of_instances', notify=False).list_items(notify=False).list_items(notify=False).trait('value'), handler=handler)
    single_value_instance = SingleValue()
    inner_list = [single_value_instance]
    container.list_of_list_of_instances.append(inner_list)
    self.assertEqual(len(events), 0)
    single_value_instance.value += 1
    event, = events
    self.assertEqual(event.object, single_value_instance)
    self.assertEqual(event.name, 'value')
    self.assertEqual(event.old, 0)
    self.assertEqual(event.new, 1)