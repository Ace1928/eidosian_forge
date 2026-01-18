from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_with_asterisk(self):
    text = 'prefix*'
    parser = traits_listener.ListenerParser(text=text)
    actual = parser.listener
    expected = traits_listener.ListenerItem(name='prefix', metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=actual)
    self.assertEqual(actual, expected)