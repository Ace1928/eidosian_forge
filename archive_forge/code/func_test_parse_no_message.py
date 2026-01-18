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
def test_parse_no_message(self):
    x = Tag()
    x.set_raw_string(self.make_tag_text(message=None))
    self.assertEqual(None, x.message)
    self.assertEqual(b'Linus Torvalds <torvalds@woody.linux-foundation.org>', x.tagger)
    self.assertEqual(datetime.datetime.utcfromtimestamp(x.tag_time), datetime.datetime(2007, 7, 1, 19, 54, 34))
    self.assertEqual(-25200, x.tag_timezone)
    self.assertEqual(b'v2.6.22-rc7', x.name)