import io
import unittest
import importlib_resources as resources
from importlib_resources._adapters import (
from . import util
def test_open_invalid_mode(self):
    with self.assertRaises(ValueError):
        self.files.open('0')