import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_suppress_deprecation_warnings(self):
    """suppress_deprecation_warnings sets DeprecationWarning to ignored."""
    symbol_versioning.suppress_deprecation_warnings()
    self.assertFirstWarning('ignore', DeprecationWarning)