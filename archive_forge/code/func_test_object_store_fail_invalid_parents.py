import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
def test_object_store_fail_invalid_parents(self):
    r = self._repo
    self.assertRaises(ObjectFormatException, r._add_graftpoints, {self._shas[-1]: ['1']})