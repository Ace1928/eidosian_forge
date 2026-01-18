import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
def test_only_once(self):
    gw = self.get_walker([b'a', b'b'], {b'a': [b'c'], b'b': [b'd'], b'c': [b'e'], b'd': [b'e'], b'e': []})
    walk = []
    acked = False
    walk.append(next(gw))
    walk.append(next(gw))
    if walk == [b'a' * 40, b'c' * 40] or walk == [b'b' * 40, b'd' * 40]:
        gw.ack(walk[0])
        acked = True
    walk.append(next(gw))
    if not acked and walk[2] == b'c' * 40:
        gw.ack(b'a' * 40)
    elif not acked and walk[2] == b'd' * 40:
        gw.ack(b'b' * 40)
    walk.append(next(gw))
    self.assertIs(None, next(gw))
    self.assertEqual([b'a' * 40, b'b' * 40, b'c' * 40, b'd' * 40], sorted(walk))
    self.assertLess(walk.index(b'a' * 40), walk.index(b'c' * 40))
    self.assertLess(walk.index(b'b' * 40), walk.index(b'd' * 40))