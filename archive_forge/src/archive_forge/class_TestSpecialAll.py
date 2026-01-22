from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
class TestSpecialAll(TestCase):
    """
    Tests for suppression of unused import warnings by C{__all__}.
    """

    def test_ignoredInFunction(self):
        """
        An C{__all__} definition does not suppress unused import warnings in a
        function scope.
        """
        self.flakes('\n        def foo():\n            import bar\n            __all__ = ["bar"]\n        ', m.UnusedImport, m.UnusedVariable)

    def test_ignoredInClass(self):
        """
        An C{__all__} definition in a class does not suppress unused import warnings.
        """
        self.flakes('\n        import bar\n        class foo:\n            __all__ = ["bar"]\n        ', m.UnusedImport)

    def test_ignored_when_not_directly_assigned(self):
        self.flakes('\n        import bar\n        (__all__,) = ("foo",)\n        ', m.UnusedImport)

    def test_warningSuppressed(self):
        """
        If a name is imported and unused but is named in C{__all__}, no warning
        is reported.
        """
        self.flakes('\n        import foo\n        __all__ = ["foo"]\n        ')
        self.flakes('\n        import foo\n        __all__ = ("foo",)\n        ')

    def test_augmentedAssignment(self):
        """
        The C{__all__} variable is defined incrementally.
        """
        self.flakes("\n        import a\n        import c\n        __all__ = ['a']\n        __all__ += ['b']\n        if 1 < 3:\n            __all__ += ['c', 'd']\n        ", m.UndefinedExport, m.UndefinedExport)

    def test_list_concatenation_assignment(self):
        """
        The C{__all__} variable is defined through list concatenation.
        """
        self.flakes("\n        import sys\n        __all__ = ['a'] + ['b'] + ['c']\n        ", m.UndefinedExport, m.UndefinedExport, m.UndefinedExport, m.UnusedImport)

    def test_tuple_concatenation_assignment(self):
        """
        The C{__all__} variable is defined through tuple concatenation.
        """
        self.flakes("\n        import sys\n        __all__ = ('a',) + ('b',) + ('c',)\n        ", m.UndefinedExport, m.UndefinedExport, m.UndefinedExport, m.UnusedImport)

    def test_all_with_attributes(self):
        self.flakes('\n        from foo import bar\n        __all__ = [bar.__name__]\n        ')

    def test_all_with_names(self):
        self.flakes('\n        from foo import bar\n        __all__ = [bar]\n        ')

    def test_all_with_attributes_added(self):
        self.flakes('\n        from foo import bar\n        from bar import baz\n        __all__ = [bar.__name__] + [baz.__name__]\n        ')

    def test_all_mixed_attributes_and_strings(self):
        self.flakes("\n        from foo import bar\n        from foo import baz\n        __all__ = ['bar', baz.__name__]\n        ")

    def test_unboundExported(self):
        """
        If C{__all__} includes a name which is not bound, a warning is emitted.
        """
        self.flakes('\n        __all__ = ["foo"]\n        ', m.UndefinedExport)
        for filename in ['foo/__init__.py', '__init__.py']:
            self.flakes('\n            __all__ = ["foo"]\n            ', filename=filename)

    def test_importStarExported(self):
        """
        Report undefined if import * is used
        """
        self.flakes("\n        from math import *\n        __all__ = ['sin', 'cos']\n        csc(1)\n        ", m.ImportStarUsed, m.ImportStarUsage, m.ImportStarUsage, m.ImportStarUsage)

    def test_importStarNotExported(self):
        """Report unused import when not needed to satisfy __all__."""
        self.flakes("\n        from foolib import *\n        a = 1\n        __all__ = ['a']\n        ", m.ImportStarUsed, m.UnusedImport)

    def test_usedInGenExp(self):
        """
        Using a global in a generator expression results in no warnings.
        """
        self.flakes('import fu; (fu for _ in range(1))')
        self.flakes('import fu; (1 for _ in range(1) if fu)')

    def test_redefinedByGenExp(self):
        """
        Re-using a global name as the loop variable for a generator
        expression results in a redefinition warning.
        """
        self.flakes('import fu; (1 for fu in range(1))', m.RedefinedWhileUnused, m.UnusedImport)

    def test_usedAsDecorator(self):
        """
        Using a global name in a decorator statement results in no warnings,
        but using an undefined name in a decorator statement results in an
        undefined name warning.
        """
        self.flakes('\n        from interior import decorate\n        @decorate\n        def f():\n            return "hello"\n        ')
        self.flakes('\n        from interior import decorate\n        @decorate(\'value\')\n        def f():\n            return "hello"\n        ')
        self.flakes('\n        @decorate\n        def f():\n            return "hello"\n        ', m.UndefinedName)

    def test_usedAsClassDecorator(self):
        """
        Using an imported name as a class decorator results in no warnings,
        but using an undefined name as a class decorator results in an
        undefined name warning.
        """
        self.flakes('\n        from interior import decorate\n        @decorate\n        class foo:\n            pass\n        ')
        self.flakes('\n        from interior import decorate\n        @decorate("foo")\n        class bar:\n            pass\n        ')
        self.flakes('\n        @decorate\n        class foo:\n            pass\n        ', m.UndefinedName)