from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_unboundExported(self):
    """
        If C{__all__} includes a name which is not bound, a warning is emitted.
        """
    self.flakes('\n        __all__ = ["foo"]\n        ', m.UndefinedExport)
    for filename in ['foo/__init__.py', '__init__.py']:
        self.flakes('\n            __all__ = ["foo"]\n            ', filename=filename)