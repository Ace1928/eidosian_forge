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
def test_tree_items_dir_sort(self):
    x = Tree()
    for name, item in _TREE_ITEMS.items():
        x[name] = item
    self.assertEqual(_SORTED_TREE_ITEMS, x.items())