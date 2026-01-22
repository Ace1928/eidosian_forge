import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
class TestHexToSha(TestCase):

    def test_simple(self):
        self.assertEqual(b'\xab\xcd' * 10, hex_to_sha(b'abcd' * 10))

    def test_reverse(self):
        self.assertEqual(b'abcd' * 10, sha_to_hex(b'\xab\xcd' * 10))