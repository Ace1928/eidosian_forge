import unittest
from bpython import keys
class TestUrwidKeys(unittest.TestCase):

    def test_keymap_map(self):
        """Verify KeyMap.map being a dictionary with the correct
        length."""
        self.assertEqual(len(keys.urwid_key_dispatch.map), 64)

    def test_keymap_setitem(self):
        """Verify keys.KeyMap correctly setting items."""
        keys.urwid_key_dispatch['simon'] = 'awesome'
        self.assertEqual(keys.urwid_key_dispatch['simon'], 'awesome')

    def test_keymap_delitem(self):
        """Verify keys.KeyMap correctly removing items."""
        keys.urwid_key_dispatch['simon'] = 'awesome'
        del keys.urwid_key_dispatch['simon']
        if 'simon' in keys.urwid_key_dispatch.map:
            raise Exception('Key still exists in dictionary')

    def test_keymap_getitem(self):
        """Verify keys.KeyMap correctly looking up items."""
        self.assertEqual(keys.urwid_key_dispatch['F11'], 'f11')
        self.assertEqual(keys.urwid_key_dispatch['C-a'], 'ctrl a')
        self.assertEqual(keys.urwid_key_dispatch['M-a'], 'meta a')

    def test_keymap_keyerror(self):
        """Verify keys.KeyMap raising KeyError when getting undefined key"""
        with self.assertRaises(KeyError):
            keys.urwid_key_dispatch['C-asdf']
            keys.urwid_key_dispatch['C-qwerty']