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
def test_tree_update_id(self):
    x = Tree()
    x[b'a.c'] = (33261, b'd80c186a03f423a81b39df39dc87fd269736ca86')
    self.assertEqual(b'0c5c6bc2c081accfbc250331b19e43b904ab9cdd', x.id)
    x[b'a.b'] = (stat.S_IFDIR, b'd80c186a03f423a81b39df39dc87fd269736ca86')
    self.assertEqual(b'07bfcb5f3ada15bbebdfa3bbb8fd858a363925c8', x.id)