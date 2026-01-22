import sys
from . import case
from . import util
def _removeTestAtIndex(self, index):
    """Stop holding a reference to the TestCase at index."""
    try:
        test = self._tests[index]
    except TypeError:
        pass
    else:
        if hasattr(test, 'countTestCases'):
            self._removed_tests += test.countTestCases()
        self._tests[index] = None