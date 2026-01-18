from collections import defaultdict
import unittest
from lazr.uri import (
def test_normalisation(self):
    self.assertEqual(str(URI('eXAMPLE://a/./b/../b/%63/%7bfoo%7d')), 'example://a/b/c/%7Bfoo%7D')
    self.assertEqual(str(URI('http://www.EXAMPLE.com/')), 'http://www.example.com/')
    self.assertEqual(str(URI('http://www.gnome.org/%7ejamesh/')), 'http://www.gnome.org/~jamesh/')
    self.assertEqual(str(URI('http://example.com')), 'http://example.com/')
    self.assertEqual(str(URI('http://example.com:/')), 'http://example.com/')
    self.assertEqual(str(URI('http://example.com:80/')), 'http://example.com/')