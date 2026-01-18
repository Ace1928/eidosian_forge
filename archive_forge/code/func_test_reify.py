import os
import unittest
from unittest.mock import patch
from kivy.utils import (boundary, escape_markup, format_bytes_to_human,
from kivy import utils
def test_reify(self):
    first = self.fib_100
    second = self.fib_100
    assert first == second