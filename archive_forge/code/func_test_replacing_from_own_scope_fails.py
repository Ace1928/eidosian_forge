import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
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