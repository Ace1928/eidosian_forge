from io import BytesIO
from .. import errors, filters
from ..filters import (ContentFilter, ContentFilterContext,
from ..osutils import sha_string
from . import TestCase, TestCaseInTempDir
class TestFilterStackMaps(TestCase):

    def _register_map(self, pref, stk1, stk2):

        def stk_lookup(key):
            return {'v1': stk1, 'v2': stk2}.get(key)
        filters.filter_stacks_registry.register(pref, stk_lookup)

    def test_filter_stack_maps(self):
        original_registry = filters._reset_registry()
        self.addCleanup(filters._reset_registry, original_registry)
        a_stack = [ContentFilter('b', 'c')]
        z_stack = [ContentFilter('y', 'x'), ContentFilter('w', 'v')]
        self._register_map('foo', a_stack, z_stack)
        self.assertEqual(['foo'], _get_registered_names())
        self._register_map('bar', z_stack, a_stack)
        self.assertEqual(['bar', 'foo'], _get_registered_names())
        self.assertRaises(KeyError, self._register_map, 'foo', [], [])

    def test_get_filter_stack_for(self):
        original_registry = filters._reset_registry()
        self.addCleanup(filters._reset_registry, original_registry)
        a_stack = [ContentFilter('b', 'c')]
        d_stack = [ContentFilter('d', 'D')]
        z_stack = [ContentFilter('y', 'x'), ContentFilter('w', 'v')]
        self._register_map('foo', a_stack, z_stack)
        self._register_map('bar', d_stack, z_stack)
        prefs = (('foo', 'v1'),)
        self.assertEqual(a_stack, _get_filter_stack_for(prefs))
        prefs = (('foo', 'v2'),)
        self.assertEqual(z_stack, _get_filter_stack_for(prefs))
        prefs = (('foo', 'v1'), ('bar', 'v1'))
        self.assertEqual(a_stack + d_stack, _get_filter_stack_for(prefs))
        prefs = (('baz', 'v1'),)
        self.assertEqual([], _get_filter_stack_for(prefs))
        prefs = (('foo', 'v3'),)
        self.assertEqual([], _get_filter_stack_for(prefs))
        prefs = (('foo', None), ('bar', 'v1'))
        self.assertEqual(d_stack, _get_filter_stack_for(prefs))