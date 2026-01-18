from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def test_parse_comma_separated_text(self):
    text = 'child1, child2, child3'
    parser = traits_listener.ListenerParser(text=text)
    listener_group = parser.listener
    common_traits = dict(metadata_name='', metadata_defined=True, is_anytrait=False, dispatch='', notify=True, is_list_handler=False, type=traits_listener.ANY_LISTENER, next=None)
    expected_items = [traits_listener.ListenerItem(name='child1', **common_traits), traits_listener.ListenerItem(name='child2', **common_traits), traits_listener.ListenerItem(name='child3', **common_traits)]
    self.assertEqual(len(listener_group.items), len(expected_items))
    for actual, expected in zip(listener_group.items, expected_items):
        self.assertEqual(actual, expected)