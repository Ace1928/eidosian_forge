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
def test_tree_serialize(self):
    blob = make_object(Blob, data=b'i am a blob')
    tree = Tree()
    tree[b'blob'] = (stat.S_IFREG, blob.id)
    with self.assert_serialization_on_change(tree):
        tree[b'blob2'] = (stat.S_IFREG, blob.id)