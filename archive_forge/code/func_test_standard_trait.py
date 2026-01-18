import copy
import pickle
import unittest
from traits.has_traits import (
from traits.ctrait import CTrait
from traits.observation.api import (
from traits.observation.exception_handling import (
from traits.traits import ForwardProperty, generic_trait
from traits.trait_types import Event, Float, Instance, Int, List, Map, Str
from traits.trait_errors import TraitError
def test_standard_trait(self):
    class_name = 'MyClass'
    bases = (object,)
    class_dict = {'attr': 'something', 'my_int': Int}
    update_traits_class_dict(class_name, bases, class_dict)
    self.assertEqual(class_dict[ListenerTraits], {})
    self.assertEqual(class_dict[InstanceTraits], {})
    self.assertEqual(len(class_dict[BaseTraits]), 1)
    self.assertEqual(len(class_dict[ClassTraits]), 1)
    self.assertIs(class_dict[BaseTraits]['my_int'], class_dict[ClassTraits]['my_int'])
    self.assertEqual(class_dict['attr'], 'something')
    self.assertNotIn('my_int', class_dict)