import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
def test_property_observe_extended_trait(self):
    instance_observe = ClassWithPropertyObservesDefault()
    handler_observe = mock.Mock()
    instance_observe.observe(handler_observe, 'extended_age')
    instance_depends_on = ClassWithPropertyDependsOnDefault()
    handler_otc = mock.Mock()
    instance_depends_on.on_trait_change(get_otc_handler(handler_otc), 'extended_age')
    instances = [instance_observe, instance_depends_on]
    handlers = [handler_observe, handler_otc]
    for instance, handler in zip(instances, handlers):
        with self.subTest(instance=instance, handler=handler):
            instance.info_with_default.age = 70
            self.assertEqual(handler.call_count, 1)
            self.assertEqual(instance.extended_age_n_calculations, 1)