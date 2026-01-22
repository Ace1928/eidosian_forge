import os
import shutil
import tempfile
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import Tree
from ..repo import MemoryRepo, Repo, parse_graftpoints, serialize_graftpoints
def assertSerialize(self, expected, graftpoints):
    self.assertEqual(sorted(expected), sorted(serialize_graftpoints(graftpoints)))