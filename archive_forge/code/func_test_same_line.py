import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_same_line(self):
    cf = self.from_file(b'[branch.foo] foo = bar\n')
    self.assertEqual(b'bar', cf.get((b'branch', b'foo'), b'foo'))