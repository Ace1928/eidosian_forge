import io
import pickle
import unittest
from unittest import mock
import weakref
from traits.api import (
from traits.trait_base import Undefined
from traits.observation.api import (
class TestHasTraitsPropertyObserves(unittest.TestCase):
    """ Tests Property notifications using 'observe', and compared the
    behavior with 'depends_on'
    """

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

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

    def test_property_observe_does_not_fire_default(self):
        instance_observe = ClassWithPropertyObservesDefault()
        handler_observe = mock.Mock()
        instance_observe.observe(handler_observe, 'extended_age')
        instance_depends_on = ClassWithPropertyDependsOnDefault()
        instance_depends_on.on_trait_change(get_otc_handler(mock.Mock()), 'extended_age')
        self.assertFalse(instance_observe.info_with_default_computed)
        self.assertTrue(instance_depends_on.info_with_default_computed)

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

    def test_property_observe_container(self):
        instance_observe = ClassWithPropertyObservesItems()
        handler_observe = mock.Mock()
        instance_observe.observe(handler_observe, 'discounted')
        instance_depends_on = ClassWithPropertyDependsOnItems()
        instance_depends_on.on_trait_change(get_otc_handler(mock.Mock()), 'discounted')
        for instance in [instance_observe, instance_depends_on]:
            with self.subTest(instance=instance):
                self.assertEqual(instance.discounted_n_calculations, 0)
                instance.list_of_infos.append(PersonInfo(age=30))
                self.assertEqual(instance.discounted_n_calculations, 1)
                instance.list_of_infos[-1].age = 80
                self.assertEqual(instance.discounted_n_calculations, 2)

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

    def test_property_with_cache(self):
        instance = ClassWithPropertyObservesWithCache()
        handler = mock.Mock()
        instance.observe(handler, 'discounted')
        instance.age = 1
        (event,), _ = handler.call_args_list[-1]
        self.assertIs(event.object, instance)
        self.assertEqual(event.name, 'discounted')
        self.assertIs(event.old, Undefined)
        self.assertIs(event.new, False)
        handler.reset_mock()
        instance.age = 80
        (event,), _ = handler.call_args_list[-1]
        self.assertIs(event.object, instance)
        self.assertEqual(event.name, 'discounted')
        self.assertIs(event.old, False)
        self.assertIs(event.new, True)

    def test_property_default_initializer_with_value_in_init(self):
        with self.assertRaises(AttributeError):
            ClassWithPropertyDependsOnInit(info_without_default=PersonInfo(age=30))
        instance = ClassWithPropertyObservesInit(info_without_default=PersonInfo(age=30))
        handler = mock.Mock()
        instance.observe(handler, 'extended_age')
        self.assertFalse(instance.sample_info_default_computed)
        self.assertEqual(instance.sample_info.age, 30)
        self.assertEqual(instance.extended_age, 30)
        self.assertEqual(handler.call_count, 0)
        instance.sample_info.age = 40
        self.assertEqual(handler.call_count, 1)
        instance_no_property = ClassWithInstanceDefaultInit(info_without_default=PersonInfo(age=30))
        self.assertFalse(instance_no_property.sample_info_default_computed)
        self.assertEqual(instance_no_property.sample_info.age, 30)

    def test_property_decorated_observer(self):
        instance_observe = ClassWithPropertyObservesDecorated(age=30)
        instance_depends_on = ClassWithPropertyDependsOnDecorated(age=30)
        for instance in [instance_observe, instance_depends_on]:
            with self.subTest(instance=instance):
                self.assertEqual(len(instance.discounted_events), 1)

    def test_garbage_collectable(self):
        instance = ClassWithPropertyObservesDefault()
        instance_ref = weakref.ref(instance)
        del instance
        self.assertIsNone(instance_ref())

    def test_property_with_no_getter(self):
        instance = ClassWithPropertyMissingGetter()
        try:
            instance.age += 1
        except Exception:
            self.fail('Having property with undefined getter/setter should not prevent the observed traits from being changed.')

    def test_property_with_missing_dependent(self):
        instance = ClassWithPropertyTraitNotFound()
        with self.assertRaises(ValueError) as exception_context:
            instance.person = PersonInfo()
        self.assertIn("Trait named 'last_name' not found", str(exception_context.exception))

    def test_pickle_has_traits_with_property_observe(self):
        instance = ClassWithPropertyMultipleObserves()
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            serialized = pickle.dumps(instance, protocol=protocol)
            deserialized = pickle.loads(serialized)
            handler = mock.Mock()
            deserialized.observe(handler, 'computed_value')
            deserialized.age = 1
            self.assertEqual(handler.call_count, 1)