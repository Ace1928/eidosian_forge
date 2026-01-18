import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_push_pop1(self):
    """Basic push/pop test"""
    q = self.queue()
    q.push(b'a')
    q.push(b'b')
    q.push(b'c')
    self.assertEqual(q.pop(), b'c')
    self.assertEqual(q.pop(), b'b')
    self.assertEqual(q.pop(), b'a')
    self.assertEqual(q.pop(), None)