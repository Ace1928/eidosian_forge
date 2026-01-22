from typing import Any
from unittest import TestCase
from .common import HyperlinkTestCase
class _ExpectedException(Exception):
    """An exception used to test HyperlinkTestCase.assertRaises."""