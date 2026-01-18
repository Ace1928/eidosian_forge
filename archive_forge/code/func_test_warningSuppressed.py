from pyflakes import messages as m
from pyflakes.checker import (
from pyflakes.test.harness import TestCase, skip
def test_warningSuppressed(self):
    """
        If a name is imported and unused but is named in C{__all__}, no warning
        is reported.
        """
    self.flakes('\n        import foo\n        __all__ = ["foo"]\n        ')
    self.flakes('\n        import foo\n        __all__ = ("foo",)\n        ')