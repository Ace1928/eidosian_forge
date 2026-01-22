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
class TestCreateTraitsMetaDict(unittest.TestCase):

    def test_class_attributes(self):
        class_name = 'MyClass'
        bases = (object,)
        class_dict = {'attr': 'something'}
        update_traits_class_dict(class_name, bases, class_dict)
        self.assertEqual(class_dict['attr'], 'something')
        for kind in (BaseTraits, ClassTraits, ListenerTraits, InstanceTraits):
            self.assertEqual(class_dict[kind], {})

    def test_forward_property(self):
        class_name = 'MyClass'
        bases = (object,)
        class_dict = {'attr': 'something', 'my_property': ForwardProperty({}), '_get_my_property': _dummy_getter, '_set_my_property': _dummy_setter, '_validate_my_property': _dummy_validator}
        update_traits_class_dict(class_name, bases, class_dict)
        self.assertEqual(class_dict[ListenerTraits], {})
        self.assertEqual(class_dict[InstanceTraits], {})
        self.assertEqual(len(class_dict[BaseTraits]), 1)
        self.assertEqual(len(class_dict[ClassTraits]), 1)
        self.assertIs(class_dict[BaseTraits]['my_property'], class_dict[ClassTraits]['my_property'])
        self.assertEqual(class_dict['attr'], 'something')
        self.assertNotIn('my_property', class_dict)

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

    def test_prefix_trait(self):
        class_name = 'MyClass'
        bases = (object,)
        class_dict = {'attr': 'something', 'my_int_': Int}
        update_traits_class_dict(class_name, bases, class_dict)
        for kind in (BaseTraits, ClassTraits, ListenerTraits, InstanceTraits):
            self.assertEqual(class_dict[kind], {})
        self.assertIn('my_int', class_dict[PrefixTraits])
        self.assertEqual(class_dict['attr'], 'something')
        self.assertNotIn('my_int', class_dict)

    def test_listener_trait(self):

        @on_trait_change('something')
        def listener(self):
            pass
        class_name = 'MyClass'
        bases = (object,)
        class_dict = {'attr': 'something', 'my_listener': listener}
        update_traits_class_dict(class_name, bases, class_dict)
        self.assertEqual(class_dict[BaseTraits], {})
        self.assertEqual(class_dict[ClassTraits], {})
        self.assertEqual(class_dict[InstanceTraits], {})
        self.assertEqual(class_dict[ListenerTraits], {'my_listener': ('method', {'pattern': 'something', 'post_init': False, 'dispatch': 'same'})})

    def test_observe_trait(self):

        @observe(trait('value'), post_init=True, dispatch='ui')
        @observe('name')
        def handler(self, event):
            pass
        class_name = 'MyClass'
        bases = (object,)
        class_dict = {'attr': 'something', 'my_listener': handler}
        update_traits_class_dict(class_name, bases, class_dict)
        self.assertEqual(class_dict[ObserverTraits], {'my_listener': [{'graphs': compile_str('name'), 'post_init': False, 'dispatch': 'same', 'handler_getter': getattr}, {'graphs': compile_expr(trait('value')), 'post_init': True, 'dispatch': 'ui', 'handler_getter': getattr}]})

    def test_python_property(self):
        class_name = 'MyClass'
        bases = (object,)
        class_dict = {'attr': 'something', 'my_property': property(_dummy_getter)}
        update_traits_class_dict(class_name, bases, class_dict)
        self.assertEqual(class_dict[BaseTraits], {})
        self.assertEqual(class_dict[InstanceTraits], {})
        self.assertEqual(class_dict[ListenerTraits], {})
        self.assertIs(class_dict[ClassTraits]['my_property'], generic_trait)

    def test_complex_baseclass(self):

        class Base(HasTraits):
            x = Int
        class_name = 'MyClass'
        bases = (Base,)
        class_dict = {'attr': 'something', 'my_trait': Float()}
        update_traits_class_dict(class_name, bases, class_dict)
        self.assertEqual(class_dict[InstanceTraits], {})
        self.assertEqual(class_dict[ListenerTraits], {})
        self.assertIs(class_dict[BaseTraits]['x'], class_dict[ClassTraits]['x'])
        self.assertIs(class_dict[BaseTraits]['my_trait'], class_dict[ClassTraits]['my_trait'])