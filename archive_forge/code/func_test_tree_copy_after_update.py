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
def test_tree_copy_after_update(self):
    """Check Tree.id is correctly updated when the tree is copied after updated."""
    shas = []
    tree = Tree()
    shas.append(tree.id)
    tree.add(b'data', 420, Blob().id)
    copied = tree.copy()
    shas.append(tree.id)
    shas.append(copied.id)
    self.assertNotIn(shas[0], shas[1:])
    self.assertEqual(shas[1], shas[2])