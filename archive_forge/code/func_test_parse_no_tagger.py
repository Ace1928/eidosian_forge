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
def test_parse_no_tagger(self):
    x = Tag()
    x.set_raw_string(self.make_tag_text(tagger=None))
    self.assertEqual(None, x.tagger)
    self.assertEqual(b'v2.6.22-rc7', x.name)
    self.assertEqual(None, x.tag_time)