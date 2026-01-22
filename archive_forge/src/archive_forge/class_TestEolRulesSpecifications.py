from .. import errors
from ..filters import _get_filter_stack_for
from ..filters.eol import _to_crlf_converter, _to_lf_converter
from . import TestCase
class TestEolRulesSpecifications(TestCase):

    def test_exact_value(self):
        """'eol = exact' should have no content filters"""
        prefs = (('eol', 'exact'),)
        self.assertEqual([], _get_filter_stack_for(prefs))

    def test_other_known_values(self):
        """These known eol values have corresponding filters."""
        known_values = ('lf', 'crlf', 'native', 'native-with-crlf-in-repo', 'lf-with-crlf-in-repo', 'crlf-with-crlf-in-repo')
        for value in known_values:
            prefs = (('eol', value),)
            self.assertNotEqual([], _get_filter_stack_for(prefs))

    def test_unknown_value(self):
        """
        Unknown eol values should raise an error.
        """
        prefs = (('eol', 'unknown-value'),)
        self.assertRaises(errors.BzrError, _get_filter_stack_for, prefs)

    def test_eol_missing_altogether_is_ok(self):
        """
        Not having eol in the set of preferences should be ok.
        """
        prefs = (('eol', None),)
        self.assertEqual([], _get_filter_stack_for(prefs))