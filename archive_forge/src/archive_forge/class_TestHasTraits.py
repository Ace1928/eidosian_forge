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
class TestHasTraits(unittest.TestCase):

    def test__class_traits(self):

        class Base(HasTraits):
            pin = Int
        a = Base()
        a_class_traits = a._class_traits()
        self.assertIsInstance(a_class_traits, dict)
        self.assertIn('pin', a_class_traits)
        self.assertIsInstance(a_class_traits['pin'], CTrait)
        b = Base()
        self.assertIs(b._class_traits(), a_class_traits)

    def test__instance_traits(self):

        class Base(HasTraits):
            pin = Int
        a = Base()
        a_instance_traits = a._instance_traits()
        self.assertIsInstance(a_instance_traits, dict)
        self.assertIs(a._instance_traits(), a_instance_traits)
        b = Base()
        self.assertIsNot(b._instance_traits(), a_instance_traits)

    def test__trait_notifications_enabled(self):

        class Base(HasTraits):
            foo = Int(0)
            foo_notify_count = Int(0)

            def _foo_changed(self):
                self.foo_notify_count += 1
        a = Base()
        self.assertTrue(a._trait_notifications_enabled())
        old_count = a.foo_notify_count
        a.foo += 1
        self.assertEqual(a.foo_notify_count, old_count + 1)
        a._trait_change_notify(False)
        self.assertFalse(a._trait_notifications_enabled())
        old_count = a.foo_notify_count
        a.foo += 1
        self.assertEqual(a.foo_notify_count, old_count)
        a._trait_change_notify(True)
        self.assertTrue(a._trait_notifications_enabled())
        old_count = a.foo_notify_count
        a.foo += 1
        self.assertEqual(a.foo_notify_count, old_count + 1)

    def test__trait_notifications_vetoed(self):

        class SomeEvent(HasTraits):
            event_id = Int()

        class Target(HasTraits):
            event = Event(Instance(SomeEvent))
            event_count = Int(0)

            def _event_fired(self):
                self.event_count += 1
        target = Target()
        event = SomeEvent(event_id=1234)
        self.assertFalse(event._trait_notifications_vetoed())
        old_count = target.event_count
        target.event = event
        self.assertEqual(target.event_count, old_count + 1)
        event._trait_veto_notify(True)
        self.assertTrue(event._trait_notifications_vetoed())
        old_count = target.event_count
        target.event = event
        self.assertEqual(target.event_count, old_count)
        event._trait_veto_notify(False)
        self.assertFalse(event._trait_notifications_vetoed())
        old_count = target.event_count
        target.event = event
        self.assertEqual(target.event_count, old_count + 1)

    def test__object_notifiers_vetoed(self):

        class SomeEvent(HasTraits):
            event_id = Int()

        class Target(HasTraits):
            event = Event(Instance(SomeEvent))
            event_count = Int(0)
        target = Target()
        event = SomeEvent(event_id=9)

        def object_handler(object, name, old, new):
            if name == 'event':
                object.event_count += 1
        target.on_trait_change(object_handler, name='anytrait')
        self.assertFalse(event._trait_notifications_vetoed())
        old_count = target.event_count
        target.event = event
        self.assertEqual(target.event_count, old_count + 1)
        event._trait_veto_notify(True)
        self.assertTrue(event._trait_notifications_vetoed())
        old_count = target.event_count
        target.event = event
        self.assertEqual(target.event_count, old_count)
        event._trait_veto_notify(False)
        self.assertFalse(event._trait_notifications_vetoed())
        old_count = target.event_count
        target.event = event
        self.assertEqual(target.event_count, old_count + 1)

    def test_traits_inited(self):
        foo = HasTraits()
        self.assertTrue(foo.traits_inited())

    def test__trait_set_inited(self):
        foo = HasTraits.__new__(HasTraits)
        self.assertFalse(foo.traits_inited())
        foo._trait_set_inited()
        self.assertTrue(foo.traits_inited())

    def test_generic_getattr_exception(self):

        class PropertyLike:
            """
            Data descriptor giving a property-like object that produces
            successive reciprocals on __get__. This means that it raises
            on first access, but not on subsequent accesses.
            """

            def __init__(self):
                self.n = 0

            def __get__(self, obj, type=None):
                old_n = self.n
                self.n += 1
                return 1 / old_n

            def __set__(self, obj, value):
                raise AttributeError('Read-only descriptor')

        class A(HasTraits):
            fruit = PropertyLike()
            banana_ = Int(1729)
        a = A()
        with self.assertRaises(ZeroDivisionError):
            a.fruit
        with self.assertRaises(AttributeError):
            a.veg
        self.assertEqual(a.banananana, 1729)

    def test_deepcopy_memoization(self):

        class A(HasTraits):
            x = Int()
            y = Str()
        a = A()
        objs = [a, a]
        objs_copy = copy.deepcopy(objs)
        self.assertIsNot(objs_copy[0], objs[0])
        self.assertIs(objs_copy[0], objs_copy[1])

    def test_add_class_trait(self):

        class A(HasTraits):
            pass
        A.add_class_trait('y', Str())
        a = A()
        self.assertEqual(a.y, '')

    def test_add_class_trait_affects_existing_instances(self):

        class A(HasTraits):
            pass
        a = A()
        A.add_class_trait('y', Str())
        self.assertEqual(a.y, '')

    def test_add_class_trait_affects_subclasses(self):

        class A(HasTraits):
            pass

        class B(A):
            pass

        class C(B):
            pass

        class D(B):
            pass
        A.add_class_trait('y', Str())
        self.assertEqual(A().y, '')
        self.assertEqual(B().y, '')
        self.assertEqual(C().y, '')
        self.assertEqual(D().y, '')

    def test_add_class_trait_has_items_and_subclasses(self):

        class A(HasTraits):
            pass

        class B(A):
            pass

        class C(B):
            pass
        A.add_class_trait('x', List(Int))
        self.assertEqual(A().x, [])
        self.assertEqual(B().x, [])
        self.assertEqual(C().x, [])
        A.add_class_trait('y', Map({'yes': 1, 'no': 0}, default_value='no'))
        self.assertEqual(A().y, 'no')
        self.assertEqual(B().y, 'no')
        self.assertEqual(C().y, 'no')

    def test_add_class_trait_add_prefix_traits(self):

        class A(HasTraits):
            pass
        A.add_class_trait('abc_', Str())
        A.add_class_trait('abc_def_', Int())
        a = A()
        self.assertEqual(a.abc_def_g, 0)
        self.assertEqual(a.abc_z, '')

    def test_add_class_trait_when_trait_already_exists(self):

        class A(HasTraits):
            foo = Int()
        with self.assertRaises(TraitError):
            A.add_class_trait('foo', List())
        self.assertEqual(A().foo, 0)
        with self.assertRaises(AttributeError):
            A().foo_items

    def test_add_class_trait_when_trait_already_exists_in_subclass(self):

        class A(HasTraits):
            pass

        class B(A):
            foo = Int()
        A.add_class_trait('foo', Str())
        self.assertEqual(A().foo, '')
        self.assertEqual(B().foo, 0)

    def test_traits_method_with_dunder_metadata(self):

        class A(HasTraits):
            foo = Int(__extension_point__=True)
            bar = Int(__extension_point__=False)
            baz = Int()
        a = A(foo=3, bar=4, baz=5)
        self.assertEqual(a.traits(__extension_point__=True), {'foo': a.trait('foo')})
        self.assertEqual(A.class_traits(__extension_point__=True), {'foo': A.class_traits()['foo']})

    def test_decorated_changed_method(self):
        events = []

        class A(HasTraits):
            foo = Int()

            @on_trait_change('foo')
            def _foo_changed(self, obj, name, old, new):
                events.append((obj, name, old, new))
        a = A()
        a.foo = 23
        self.assertEqual(events, [(a, 'foo', 0, 23)])

    def test_observed_changed_method(self):
        events = []

        class A(HasTraits):
            foo = Int()

            @observe('foo')
            def _foo_changed(self, event):
                events.append(event)
        a = A()
        a.foo = 23
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.object, a)
        self.assertEqual(event.name, 'foo')
        self.assertEqual(event.old, 0)
        self.assertEqual(event.new, 23)

    def test_decorated_changed_method_subclass(self):
        events = []

        class A(HasTraits):
            foo = Int()

            @on_trait_change('foo')
            def _foo_changed(self, obj, name, old, new):
                events.append((obj, name, old, new))

        class B(A):
            pass
        a = B()
        a.foo = 23
        self.assertEqual(events, [(a, 'foo', 0, 23)])