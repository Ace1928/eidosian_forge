import warnings
from breezy import symbol_versioning
from breezy.symbol_versioning import (deprecated_function, deprecated_in,
from breezy.tests import TestCase
class TestSuppressAndActivate(TestCase):

    def setUp(self):
        super().setUp()
        existing_filters = list(warnings.filters)

        def restore():
            warnings.filters[:] = existing_filters
        self.addCleanup(restore)
        warnings.resetwarnings()

    def assertFirstWarning(self, action, category):
        """Test the first warning in the filters is correct"""
        first = warnings.filters[0]
        self.assertEqual((action, category), (first[0], first[2]))

    def test_suppress_deprecation_warnings(self):
        """suppress_deprecation_warnings sets DeprecationWarning to ignored."""
        symbol_versioning.suppress_deprecation_warnings()
        self.assertFirstWarning('ignore', DeprecationWarning)

    def test_set_restore_filters(self):
        original_filters = warnings.filters[:]
        symbol_versioning.suppress_deprecation_warnings()()
        self.assertEqual(original_filters, warnings.filters)

    def test_suppress_deprecation_with_warning_filter(self):
        """don't suppress if we already have a filter"""
        warnings.filterwarnings('error', category=Warning)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.suppress_deprecation_warnings(override=False)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))

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

    def test_activate_deprecation_no_error(self):
        symbol_versioning.activate_deprecation_warnings()
        self.assertFirstWarning('default', DeprecationWarning)

    def test_activate_deprecation_with_error(self):
        warnings.filterwarnings('error', category=Warning)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.activate_deprecation_warnings(override=False)
        self.assertFirstWarning('error', Warning)
        self.assertEqual(1, len(warnings.filters))

    def test_activate_deprecation_with_DW_error(self):
        warnings.filterwarnings('error', category=DeprecationWarning)
        self.assertFirstWarning('error', DeprecationWarning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.activate_deprecation_warnings(override=False)
        self.assertFirstWarning('error', DeprecationWarning)
        self.assertEqual(1, len(warnings.filters))
        symbol_versioning.activate_deprecation_warnings(override=True)
        self.assertFirstWarning('default', DeprecationWarning)
        self.assertEqual(2, len(warnings.filters))