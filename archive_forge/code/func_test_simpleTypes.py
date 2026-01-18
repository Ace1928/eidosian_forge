from __future__ import annotations
import copyreg
import io
import pickle
import sys
import textwrap
from typing import Any, Callable, List, Tuple
from typing_extensions import NoReturn
from twisted.persisted import aot, crefutil, styles
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_simpleTypes(self) -> None:
    obj = (1, 2.0, 3j, True, slice(1, 2, 3), 'hello', 'world', sys.maxsize + 1, None, Ellipsis)
    rtObj = aot.unjellyFromSource(aot.jellyToSource(obj))
    self.assertEqual(obj, rtObj)