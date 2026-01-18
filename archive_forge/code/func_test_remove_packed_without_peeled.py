import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
def test_remove_packed_without_peeled(self):
    refs_file = os.path.join(self._repo.path, 'packed-refs')
    f = GitFile(refs_file)
    refs_data = f.read()
    f.close()
    f = GitFile(refs_file, 'wb')
    f.write(b'\n'.join((line for line in refs_data.split(b'\n') if not line or line[0] not in b'#^')))
    f.close()
    self._repo = Repo(self._repo.path)
    refs = self._repo.refs
    self.assertTrue(refs.remove_if_equals(b'refs/heads/packed', b'42d06bd4b77fed026b154d16493e5deab78f02ec'))