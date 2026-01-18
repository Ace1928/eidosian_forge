import os
import sys
from io import BytesIO
from unittest import skipIf
from unittest.mock import patch
from dulwich.tests import TestCase
from ..config import (
def test_sections(self):
    cd = ConfigDict()
    cd.set((b'core2',), b'foo', b'bloe')
    self.assertEqual([(b'core2',)], list(cd.sections()))