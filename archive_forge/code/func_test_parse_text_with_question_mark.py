from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_text_with_question_mark(self):
    text = 'foo?.bar?'
    parser = traits_listener.ListenerParser(text=text)
    listener = parser.listener
    self.assertEqual(listener.name, 'foo?')
    listener = listener.next
    self.assertEqual(listener.name, 'bar?')