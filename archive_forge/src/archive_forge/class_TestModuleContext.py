import re
import sys
import textwrap
from io import StringIO
from .. import commands, export_pot, option, registry, tests
class TestModuleContext(tests.TestCase):
    """Checks for source context tracking objects"""

    def check_context(self, context, path, lineno):
        self.assertEqual((context.path, context.lineno), (path, lineno))

    def test___init__(self):
        context = export_pot._ModuleContext('one.py')
        self.check_context(context, 'one.py', 1)
        context = export_pot._ModuleContext('two.py', 5)
        self.check_context(context, 'two.py', 5)

    def test_from_class(self):
        """New context returned with lineno updated from class"""
        path = 'cls.py'

        class A:
            pass

        class B:
            pass
        cls_lines = {'A': 5, 'B': 7}
        context = export_pot._ModuleContext(path, _source_info=(cls_lines, {}))
        contextA = context.from_class(A)
        self.check_context(contextA, path, 5)
        contextB1 = context.from_class(B)
        self.check_context(contextB1, path, 7)
        contextB2 = contextA.from_class(B)
        self.check_context(contextB2, path, 7)
        self.check_context(context, path, 1)
        self.assertEqual('', self.get_log())

    def test_from_class_missing(self):
        """When class has no lineno the old context details are returned"""
        path = 'cls_missing.py'

        class A:
            pass

        class M:
            pass
        context = export_pot._ModuleContext(path, 3, ({'A': 15}, {}))
        contextA = context.from_class(A)
        contextM1 = context.from_class(M)
        self.check_context(contextM1, path, 3)
        contextM2 = contextA.from_class(M)
        self.check_context(contextM2, path, 15)
        self.assertContainsRe(self.get_log(), "Definition of <.*M'> not found")

    def test_from_string(self):
        """New context returned with lineno updated from string"""
        path = 'str.py'
        str_lines = {'one': 14, 'two': 42}
        context = export_pot._ModuleContext(path, _source_info=({}, str_lines))
        context1 = context.from_string('one')
        self.check_context(context1, path, 14)
        context2A = context.from_string('two')
        self.check_context(context2A, path, 42)
        context2B = context1.from_string('two')
        self.check_context(context2B, path, 42)
        self.check_context(context, path, 1)
        self.assertEqual('', self.get_log())

    def test_from_string_missing(self):
        """When string has no lineno the old context details are returned"""
        path = 'str_missing.py'
        context = export_pot._ModuleContext(path, 4, ({}, {'line\n': 21}))
        context1 = context.from_string('line\n')
        context2A = context.from_string('not there')
        self.check_context(context2A, path, 4)
        context2B = context1.from_string('not there')
        self.check_context(context2B, path, 21)
        self.assertContainsRe(self.get_log(), "String b?'not there' not found")