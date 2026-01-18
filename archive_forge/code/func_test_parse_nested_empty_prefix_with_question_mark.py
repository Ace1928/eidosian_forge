from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_nested_empty_prefix_with_question_mark(self):
    text = 'foo.?'
    with self.assertRaises(TraitError) as exception_context:
        traits_listener.ListenerParser(text=text)
    self.assertIn('Expected non-empty name', str(exception_context.exception))