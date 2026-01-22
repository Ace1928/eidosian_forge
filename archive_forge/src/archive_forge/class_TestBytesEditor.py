import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
@requires_traitsui
class TestBytesEditor(unittest.TestCase):

    def test_bytes_editor_default(self):
        editor = bytes_editor()
        self.assertIsInstance(editor, traitsui.api.TextEditor)
        self.assertTrue(editor.auto_set)
        self.assertFalse(editor.enter_set)
        formatted = editor.format_func(b'\xde\xad\xbe\xef')
        self.assertEqual(formatted, 'deadbeef')
        evaluated = editor.evaluate('deadbeef')
        self.assertEqual(evaluated, b'\xde\xad\xbe\xef')

    def test_bytes_editor_options(self):
        editor = bytes_editor(auto_set=False, enter_set=True, encoding='ascii')
        self.assertIsInstance(editor, traitsui.api.TextEditor)
        self.assertFalse(editor.auto_set)
        self.assertTrue(editor.enter_set)
        formatted = editor.format_func(b'deadbeef')
        self.assertEqual(formatted, 'deadbeef')
        evaluated = editor.evaluate('deadbeef')
        self.assertEqual(evaluated, b'deadbeef')