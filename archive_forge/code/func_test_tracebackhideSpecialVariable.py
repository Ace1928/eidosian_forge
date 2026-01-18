from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_tracebackhideSpecialVariable(self):
    """
        Do not warn about unused local variable __tracebackhide__, which is
        a special variable for py.test.
        """
    self.flakes('\n            def helper():\n                __tracebackhide__ = True\n        ')