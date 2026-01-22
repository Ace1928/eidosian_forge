import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestScopeReplacer(TestCase):
    """Test the ability of the replacer to put itself into the correct scope.

    In these tests we use the global scope, because we cannot replace
    variables in the local scope. This means that we need to be careful
    and not have the replacing objects use the same name, or we would
    get collisions.
    """

    def setUp(self):
        super().setUp()
        orig_proxy = lazy_import.ScopeReplacer._should_proxy

        def restore():
            lazy_import.ScopeReplacer._should_proxy = orig_proxy
        lazy_import.ScopeReplacer._should_proxy = False

    def test_object(self):
        """ScopeReplacer can create an instance in local scope.

        An object should appear in globals() by constructing a ScopeReplacer,
        and it will be replaced with the real object upon the first request.
        """
        actions = []
        InstrumentedReplacer.use_actions(actions)
        TestClass.use_actions(actions)

        def factory(replacer, scope, name):
            actions.append('factory')
            return TestClass()
        try:
            test_obj1
        except NameError:
            pass
        else:
            self.fail('test_obj1 was not supposed to exist yet')
        InstrumentedReplacer(scope=globals(), name='test_obj1', factory=factory)
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj1, '__class__'))
        self.assertEqual('foo', test_obj1.foo(1))
        self.assertIsInstance(test_obj1, TestClass)
        self.assertEqual('foo', test_obj1.foo(2))
        self.assertEqual([('__getattribute__', 'foo'), 'factory', 'init', ('foo', 1), ('foo', 2)], actions)

    def test_setattr_replaces(self):
        """ScopeReplacer can create an instance in local scope.

        An object should appear in globals() by constructing a ScopeReplacer,
        and it will be replaced with the real object upon the first request.
        """
        actions = []
        TestClass.use_actions(actions)

        def factory(replacer, scope, name):
            return TestClass()
        try:
            test_obj6
        except NameError:
            pass
        else:
            self.fail('test_obj6 was not supposed to exist yet')
        lazy_import.ScopeReplacer(scope=globals(), name='test_obj6', factory=factory)
        self.assertEqual(lazy_import.ScopeReplacer, object.__getattribute__(test_obj6, '__class__'))
        test_obj6.bar = 'test'
        self.assertNotEqual(lazy_import.ScopeReplacer, object.__getattribute__(test_obj6, '__class__'))
        self.assertEqual('test', test_obj6.bar)

    def test_replace_side_effects(self):
        """Creating a new object should only create one entry in globals.

        And only that entry even after replacement.
        """
        try:
            test_scope1
        except NameError:
            pass
        else:
            self.fail('test_scope1 was not supposed to exist yet')
        TestClass.use_actions([])

        def factory(replacer, scope, name):
            return TestClass()
        orig_globals = set(globals().keys())
        lazy_import.ScopeReplacer(scope=globals(), name='test_scope1', factory=factory)
        new_globals = set(globals().keys())
        self.assertEqual(lazy_import.ScopeReplacer, object.__getattribute__(test_scope1, '__class__'))
        self.assertEqual('foo', test_scope1.foo(1))
        self.assertIsInstance(test_scope1, TestClass)
        final_globals = set(globals().keys())
        self.assertEqual({'test_scope1'}, new_globals - orig_globals)
        self.assertEqual(set(), orig_globals - new_globals)
        self.assertEqual(set(), final_globals - new_globals)
        self.assertEqual(set(), new_globals - final_globals)

    def test_class(self):
        actions = []
        InstrumentedReplacer.use_actions(actions)
        TestClass.use_actions(actions)

        def factory(replacer, scope, name):
            actions.append('factory')
            return TestClass
        try:
            test_class1
        except NameError:
            pass
        else:
            self.fail('test_class1 was not supposed to exist yet')
        InstrumentedReplacer(scope=globals(), name='test_class1', factory=factory)
        self.assertEqual('class_member', test_class1.class_member)
        self.assertEqual(test_class1, TestClass)
        self.assertEqual([('__getattribute__', 'class_member'), 'factory'], actions)

    def test_call_class(self):
        actions = []
        InstrumentedReplacer.use_actions(actions)
        TestClass.use_actions(actions)

        def factory(replacer, scope, name):
            actions.append('factory')
            return TestClass
        try:
            test_class2
        except NameError:
            pass
        else:
            self.fail('test_class2 was not supposed to exist yet')
        InstrumentedReplacer(scope=globals(), name='test_class2', factory=factory)
        self.assertFalse(test_class2 is TestClass)
        obj = test_class2()
        self.assertIs(test_class2, TestClass)
        self.assertIsInstance(obj, TestClass)
        self.assertEqual('class_member', obj.class_member)
        self.assertEqual([('__call__', (), {}), 'factory', 'init'], actions)

    def test_call_func(self):
        actions = []
        InstrumentedReplacer.use_actions(actions)

        def func(a, b, c=None):
            actions.append('func')
            return (a, b, c)

        def factory(replacer, scope, name):
            actions.append('factory')
            return func
        try:
            test_func1
        except NameError:
            pass
        else:
            self.fail('test_func1 was not supposed to exist yet')
        InstrumentedReplacer(scope=globals(), name='test_func1', factory=factory)
        self.assertFalse(test_func1 is func)
        val = test_func1(1, 2, c='3')
        self.assertIs(test_func1, func)
        self.assertEqual((1, 2, '3'), val)
        self.assertEqual([('__call__', (1, 2), {'c': '3'}), 'factory', 'func'], actions)

    def test_other_variable(self):
        """Test when a ScopeReplacer is assigned to another variable.

        This test could be updated if we find a way to trap '=' rather
        than just giving a belated exception.
        ScopeReplacer only knows about the variable it was created as,
        so until the object is replaced, it is illegal to pass it to
        another variable. (Though discovering this may take a while)
        """
        actions = []
        InstrumentedReplacer.use_actions(actions)
        TestClass.use_actions(actions)

        def factory(replacer, scope, name):
            actions.append('factory')
            return TestClass()
        try:
            test_obj2
        except NameError:
            pass
        else:
            self.fail('test_obj2 was not supposed to exist yet')
        InstrumentedReplacer(scope=globals(), name='test_obj2', factory=factory)
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj2, '__class__'))
        test_obj3 = test_obj2
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj2, '__class__'))
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj3, '__class__'))
        self.assertEqual('foo', test_obj3.foo(1))
        self.assertEqual(TestClass, object.__getattribute__(test_obj2, '__class__'))
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj3, '__class__'))
        self.assertEqual('foo', test_obj2.foo(2))
        self.assertEqual('foo', test_obj2.foo(3))
        self.assertRaises(lazy_import.IllegalUseOfScopeReplacer, getattr, test_obj3, 'foo')
        self.assertEqual([('__getattribute__', 'foo'), 'factory', 'init', ('foo', 1), ('foo', 2), ('foo', 3), ('__getattribute__', 'foo')], actions)

    def test_enable_proxying(self):
        """Test that we can allow ScopeReplacer to proxy."""
        actions = []
        InstrumentedReplacer.use_actions(actions)
        TestClass.use_actions(actions)

        def factory(replacer, scope, name):
            actions.append('factory')
            return TestClass()
        try:
            test_obj4
        except NameError:
            pass
        else:
            self.fail('test_obj4 was not supposed to exist yet')
        lazy_import.ScopeReplacer._should_proxy = True
        InstrumentedReplacer(scope=globals(), name='test_obj4', factory=factory)
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj4, '__class__'))
        test_obj5 = test_obj4
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj4, '__class__'))
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj5, '__class__'))
        self.assertEqual('foo', test_obj4.foo(1))
        self.assertEqual(TestClass, object.__getattribute__(test_obj4, '__class__'))
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj5, '__class__'))
        self.assertEqual('foo', test_obj4.foo(2))
        self.assertEqual('foo', test_obj5.foo(3))
        self.assertEqual('foo', test_obj5.foo(4))
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj5, '__class__'))
        self.assertEqual([('__getattribute__', 'foo'), 'factory', 'init', ('foo', 1), ('foo', 2), ('__getattribute__', 'foo'), ('foo', 3), ('__getattribute__', 'foo'), ('foo', 4)], actions)

    def test_replacing_from_own_scope_fails(self):
        """If a ScopeReplacer tries to replace itself a nice error is given"""
        actions = []
        InstrumentedReplacer.use_actions(actions)
        TestClass.use_actions(actions)

        def factory(replacer, scope, name):
            actions.append('factory')
            return scope[name]
        try:
            test_obj7
        except NameError:
            pass
        else:
            self.fail('test_obj7 was not supposed to exist yet')
        InstrumentedReplacer(scope=globals(), name='test_obj7', factory=factory)
        self.assertEqual(InstrumentedReplacer, object.__getattribute__(test_obj7, '__class__'))
        e = self.assertRaises(lazy_import.IllegalUseOfScopeReplacer, test_obj7)
        self.assertIn('replace itself', e.msg)
        self.assertEqual([('__call__', (), {}), 'factory'], actions)