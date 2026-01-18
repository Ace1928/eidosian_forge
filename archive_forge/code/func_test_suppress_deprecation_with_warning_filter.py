import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
def test_suppress_deprecation_with_warning_filter(self):
    """don't suppress if we already have a filter"""
    warnings.filterwarnings('error', category=Warning)
    self.assertFirstWarning('error', Warning)
    self.assertEqual(1, len(warnings.filters))
    symbol_versioning.suppress_deprecation_warnings(override=False)
    self.assertFirstWarning('error', Warning)
    self.assertEqual(1, len(warnings.filters))