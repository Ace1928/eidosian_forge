import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_password_editor_default(self):
    editor = password_editor()
    self.assertIsInstance(editor, traitsui.api.TextEditor)
    self.assertTrue(editor.password)
    self.assertTrue(editor.auto_set)
    self.assertFalse(editor.enter_set)