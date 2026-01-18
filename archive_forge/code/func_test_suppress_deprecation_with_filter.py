import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_suppress_deprecation_with_filter(self):
    """don't suppress if we already have a filter"""
    warnings.filterwarnings('error', category=DeprecationWarning)
    self.assertFirstWarning('error', DeprecationWarning)
    self.assertEqual(1, len(warnings.filters))
    symbol_versioning.suppress_deprecation_warnings(override=False)
    self.assertFirstWarning('error', DeprecationWarning)
    self.assertEqual(1, len(warnings.filters))
    symbol_versioning.suppress_deprecation_warnings(override=True)
    self.assertFirstWarning('ignore', DeprecationWarning)
    self.assertEqual(2, len(warnings.filters))