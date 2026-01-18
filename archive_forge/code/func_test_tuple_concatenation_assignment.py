from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_tuple_concatenation_assignment(self):
    """
        The C{__all__} variable is defined through tuple concatenation.
        """
    self.flakes("\n        import sys\n        __all__ = ('a',) + ('b',) + ('c',)\n        ", m.UndefinedExport, m.UndefinedExport, m.UndefinedExport, m.UnusedImport)