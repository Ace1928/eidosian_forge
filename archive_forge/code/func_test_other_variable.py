import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
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