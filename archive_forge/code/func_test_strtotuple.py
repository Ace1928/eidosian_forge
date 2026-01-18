import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_strtotuple(self):
    self.assertRaises(Exception, strtotuple, 'adis!_m%*+-=|')
    self.assertRaises(Exception, strtotuple, '((12, 8, 473)')
    self.assertRaises(Exception, strtotuple, '[12, 8, 473]]')
    self.assertRaises(Exception, strtotuple, '128473')
    actual = strtotuple('(12, 8, 473)')
    expected = (12, 8, 473)
    self.assertEqual(actual, expected)