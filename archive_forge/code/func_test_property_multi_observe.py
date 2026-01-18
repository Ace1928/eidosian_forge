import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_multi_observe(self):
    instance_observe = ClassWithPropertyMultipleObserves()
    handler_observe = mock.Mock()
    instance_observe.observe(handler_observe, 'computed_value')
    self.assertEqual(instance_observe.computed_value_n_calculations, 0)
    instance_depends_on = ClassWithPropertyMultipleDependsOn()
    instance_depends_on.on_trait_change(get_otc_handler(mock.Mock()), 'computed_value')
    self.assertEqual(instance_depends_on.computed_value_n_calculations, 0)
    for instance in [instance_observe, instance_depends_on]:
        with self.subTest(instance=instance):
            instance.age = 1
            self.assertEqual(instance.computed_value_n_calculations, 1)
            instance.gender = 'male'
            self.assertEqual(instance.computed_value_n_calculations, 2)