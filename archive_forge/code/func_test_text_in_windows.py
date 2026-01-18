import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
def test_text_in_windows(self):
    e1 = b'\r\n'
    q = self.queue()
    q.push(e1)
    q.close()
    q = self.queue()
    e2 = q.pop()
    self.assertEqual(e1, e2)