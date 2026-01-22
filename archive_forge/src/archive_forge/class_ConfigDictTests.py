import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
class ConfigDictTests(TestCase):

    def test_get_set(self):
        cd = ConfigDict()
        self.assertRaises(KeyError, cd.get, b'foo', b'core')
        cd.set((b'core',), b'foo', b'bla')
        self.assertEqual(b'bla', cd.get((b'core',), b'foo'))
        cd.set((b'core',), b'foo', b'bloe')
        self.assertEqual(b'bloe', cd.get((b'core',), b'foo'))

    def test_get_boolean(self):
        cd = ConfigDict()
        cd.set((b'core',), b'foo', b'true')
        self.assertTrue(cd.get_boolean((b'core',), b'foo'))
        cd.set((b'core',), b'foo', b'false')
        self.assertFalse(cd.get_boolean((b'core',), b'foo'))
        cd.set((b'core',), b'foo', b'invalid')
        self.assertRaises(ValueError, cd.get_boolean, (b'core',), b'foo')

    def test_dict(self):
        cd = ConfigDict()
        cd.set((b'core',), b'foo', b'bla')
        cd.set((b'core2',), b'foo', b'bloe')
        self.assertEqual([(b'core',), (b'core2',)], list(cd.keys()))
        self.assertEqual(cd[b'core',], {b'foo': b'bla'})
        cd[b'a'] = b'b'
        self.assertEqual(cd[b'a'], b'b')

    def test_items(self):
        cd = ConfigDict()
        cd.set((b'core',), b'foo', b'bla')
        cd.set((b'core2',), b'foo', b'bloe')
        self.assertEqual([(b'foo', b'bla')], list(cd.items((b'core',))))

    def test_items_nonexistant(self):
        cd = ConfigDict()
        cd.set((b'core2',), b'foo', b'bloe')
        self.assertEqual([], list(cd.items((b'core',))))

    def test_sections(self):
        cd = ConfigDict()
        cd.set((b'core2',), b'foo', b'bloe')
        self.assertEqual([(b'core2',)], list(cd.sections()))