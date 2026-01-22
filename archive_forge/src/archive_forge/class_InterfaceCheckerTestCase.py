import unittest
import warnings
from traits.adaptation.api import reset_global_adaptation_manager
from traits.api import (
from traits.interface_checker import InterfaceError, check_implements
from traits import has_traits
class InterfaceCheckerTestCase(unittest.TestCase):
    """ Tests to help find out if we can do type-safe casting. """

    def setUp(self):
        """ Prepares the test fixture before each test method is called. """
        self._old_check_interfaces = has_traits.CHECK_INTERFACES
        has_traits.CHECK_INTERFACES = 0
        reset_global_adaptation_manager()

    def tearDown(self):
        has_traits.CHECK_INTERFACES = self._old_check_interfaces

    def test_non_traits_class(self):
        """ non-traits class """

        class IFoo(Interface):

            def foo(self):
                pass

        @provides(IFoo)
        class Foo(object):

            def foo(self):
                pass
        check_implements(Foo, IFoo, 2)

    def test_single_interface(self):
        """ single interface """

        class IFoo(Interface):
            x = Int

        @provides(IFoo)
        class Foo(HasTraits):
            x = Int
        check_implements(Foo, IFoo, 2)

    def test_single_interface_with_invalid_method_signature(self):
        """ single interface with invalid method signature """

        class IFoo(Interface):

            def foo(self):
                pass

        @provides(IFoo)
        class Foo(HasTraits):

            def foo(self, x):
                pass
        self.assertRaises(InterfaceError, check_implements, Foo, IFoo, 2)

    def test_single_interface_with_missing_trait(self):
        """ single interface with missing trait """

        class IFoo(Interface):
            x = Int

        @provides(IFoo)
        class Foo(HasTraits):
            pass
        self.assertRaises(InterfaceError, check_implements, Foo, IFoo, 2)

    def test_single_interface_with_missing_method(self):
        """ single interface with missing method """

        class IFoo(Interface):

            def method(self):
                pass

        @provides(IFoo)
        class Foo(HasTraits):
            pass
        self.assertRaises(InterfaceError, check_implements, Foo, IFoo, 2)

    def test_multiple_interfaces(self):
        """ multiple interfaces """

        class IFoo(Interface):
            x = Int

        class IBar(Interface):
            y = Int

        class IBaz(Interface):
            z = Int

        @provides(IFoo, IBar, IBaz)
        class Foo(HasTraits):
            x = Int
            y = Int
            z = Int
        check_implements(Foo, [IFoo, IBar, IBaz], 2)

    def test_multiple_interfaces_with_invalid_method_signature(self):
        """ multiple interfaces with invalid method signature """

        class IFoo(Interface):

            def foo(self):
                pass

        class IBar(Interface):

            def bar(self):
                pass

        class IBaz(Interface):

            def baz(self):
                pass

        @provides(IFoo, IBar, IBaz)
        class Foo(HasTraits):

            def foo(self):
                pass

            def bar(self):
                pass

            def baz(self, x):
                pass
        self.assertRaises(InterfaceError, check_implements, Foo, [IFoo, IBar, IBaz], 2)

    def test_multiple_interfaces_with_missing_trait(self):
        """ multiple interfaces with missing trait """

        class IFoo(Interface):
            x = Int

        class IBar(Interface):
            y = Int

        class IBaz(Interface):
            z = Int

        @provides(IFoo, IBar, IBaz)
        class Foo(HasTraits):
            x = Int
            y = Int
        self.assertRaises(InterfaceError, check_implements, Foo, [IFoo, IBar, IBaz], 2)

    def test_multiple_interfaces_with_missing_method(self):
        """ multiple interfaces with missing method """

        class IFoo(Interface):

            def foo(self):
                pass

        class IBar(Interface):

            def bar(self):
                pass

        class IBaz(Interface):

            def baz(self):
                pass

        @provides(IFoo, IBar, IBaz)
        class Foo(HasTraits):

            def foo(self):
                pass

            def bar(self):
                pass
        self.assertRaises(InterfaceError, check_implements, Foo, [IFoo, IBar, IBaz], 2)

    def test_inherited_interfaces(self):
        """ inherited interfaces """

        class IFoo(Interface):
            x = Int

        class IBar(IFoo):
            y = Int

        class IBaz(IBar):
            z = Int

        @provides(IBaz)
        class Foo(HasTraits):
            x = Int
            y = Int
            z = Int
        check_implements(Foo, IBaz, 2)

    def test_inherited_interfaces_with_invalid_method_signature(self):
        """ inherited with invalid method signature """

        class IFoo(Interface):

            def foo(self):
                pass

        class IBar(IFoo):

            def bar(self):
                pass

        class IBaz(IBar):

            def baz(self):
                pass

        @provides(IBaz)
        class Foo(HasTraits):

            def foo(self):
                pass

            def bar(self):
                pass

            def baz(self, x):
                pass
        self.assertRaises(InterfaceError, check_implements, Foo, IBaz, 2)

    def test_inherited_interfaces_with_missing_trait(self):
        """ inherited interfaces with missing trait """

        class IFoo(Interface):
            x = Int

        class IBar(IFoo):
            y = Int

        class IBaz(IBar):
            z = Int

        @provides(IBaz)
        class Foo(HasTraits):
            x = Int
            y = Int
        self.assertRaises(InterfaceError, check_implements, Foo, IBaz, 2)

    def test_inherited_interfaces_with_missing_method(self):
        """ inherited interfaces with missing method """

        class IFoo(Interface):

            def foo(self):
                pass

        class IBar(IFoo):

            def bar(self):
                pass

        class IBaz(IBar):

            def baz(self):
                pass

        @provides(IBaz)
        class Foo(HasTraits):

            def foo(self):
                pass

            def bar(self):
                pass
        self.assertRaises(InterfaceError, check_implements, Foo, IBaz, 2)

    def test_subclasses_with_wrong_signature_methods(self):
        """ Subclasses with incorrect method signatures """

        class IFoo(Interface):

            def foo(self, argument):
                pass

        @provides(IFoo)
        class Foo(HasTraits):

            def foo(self, argument):
                pass

        class Bar(Foo):

            def foo(self):
                pass
        self.assertRaises(InterfaceError, check_implements, Bar, IFoo, 2)

    def test_instance(self):
        """ instance """

        class IFoo(Interface):
            pass

        @provides(IFoo)
        class Foo(HasTraits):
            pass

        class Bar(HasTraits):
            foo = Instance(IFoo)
        Bar(foo=Foo())

    def test_callable(self):
        """ callable """

        class IFoo(Interface):
            pass

        @provides(IFoo)
        class Foo(HasTraits):
            pass
        f = Foo()
        with warnings.catch_warnings(record=True) as warn_msgs:
            warnings.simplefilter('always', DeprecationWarning)
            self.assertEqual(f, IFoo(f))
        self.assertEqual(len(warn_msgs), 1)
        warn_msg = warn_msgs[0]
        self.assertIn('use "adapt(adaptee, protocol)" instead', str(warn_msg.message))
        self.assertIn('test_interface_checker', warn_msg.filename)

    def test_adaptation(self):
        """ adaptation """

        class IFoo(Interface):
            pass

        class Foo(HasTraits):
            pass

        @provides(IFoo)
        class FooToIFooAdapter(Adapter):
            pass
        register_factory(FooToIFooAdapter, Foo, IFoo)
        f = Foo()
        with warnings.catch_warnings(record=True) as warn_msgs:
            warnings.simplefilter('always', DeprecationWarning)
            i_foo = IFoo(f)
        self.assertNotEqual(None, i_foo)
        self.assertEqual(FooToIFooAdapter, type(i_foo))
        self.assertEqual(len(warn_msgs), 1)
        warn_msg = warn_msgs[0]
        self.assertIn('use "adapt(adaptee, protocol)" instead', str(warn_msg.message))
        self.assertIn('test_interface_checker', warn_msg.filename)