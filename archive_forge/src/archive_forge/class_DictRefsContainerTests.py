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
class DictRefsContainerTests(RefsContainerTests, TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self._refs = DictRefsContainer(dict(_TEST_REFS))

    def test_invalid_refname(self):
        self._refs._refs[b'refs/stash'] = b'00' * 20
        expected_refs = dict(_TEST_REFS)
        del expected_refs[b'refs/heads/loop']
        expected_refs[b'refs/stash'] = b'00' * 20
        self.assertEqual(expected_refs, self._refs.as_dict())