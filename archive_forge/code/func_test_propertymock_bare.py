import inspect
import time
import types
import unittest
from unittest.mock import (
from datetime import datetime
from functools import partial
def test_propertymock_bare(self):
    m = MagicMock()
    p = PropertyMock()
    type(m).foo = p
    returned = m.foo
    p.assert_called_once_with()
    self.assertIsInstance(returned, MagicMock)
    self.assertNotIsInstance(returned, PropertyMock)