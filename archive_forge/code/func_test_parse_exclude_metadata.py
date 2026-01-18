from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_exclude_metadata(self):
    text = '-foo'
    parser = traits_listener.ListenerParser(text=text)
    expected = traits_listener.ListenerItem(name='*', metadata_name='foo', metadata_defined=False, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
    self.assertEqual(parser.listener, expected)