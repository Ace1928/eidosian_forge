import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_old_value_uncached(self):
    instance = ClassWithPropertyMultipleObserves()
    handler = mock.Mock()
    instance.observe(handler, 'computed_value')
    instance.age = 1
    ((event,), _), = handler.call_args_list
    self.assertIs(event.object, instance)
    self.assertEqual(event.name, 'computed_value')
    self.assertIs(event.old, Undefined)
    self.assertIs(event.new, 1)
    handler.reset_mock()
    instance.gender = 'male'
    ((event,), _), = handler.call_args_list
    self.assertIs(event.object, instance)
    self.assertEqual(event.name, 'computed_value')
    self.assertIs(event.old, Undefined)
    self.assertIs(event.new, 5)