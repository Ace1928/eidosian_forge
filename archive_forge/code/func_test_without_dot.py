import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
def test_without_dot(self):
    self.assertAccess('Object|')
    self.assertAccess('Object|.')
    self.assertAccess('|Object.')