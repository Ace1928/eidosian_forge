import gzip
import os
import time
from io import BytesIO
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.repo import Repo as GitRepo
from ...branch import Branch
from ...controldir import BranchReferenceLoop, ControlDir
from ...errors import (ConnectionReset, DivergedBranches, NoSuchTag,
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ExecutableFeature
from ...urlutils import join as urljoin
from ..mapping import default_mapping
from ..remote import (GitRemoteRevisionTree, GitSmartRemoteNotSupported,
from ..tree import MissingNestedTree
class TestRemoteGitBranchFormat(TestCase):

    def setUp(self):
        super().setUp()
        self.format = RemoteGitBranchFormat()

    def test_get_format_description(self):
        self.assertEqual('Remote Git Branch', self.format.get_format_description())

    def test_get_network_name(self):
        self.assertEqual(b'git', self.format.network_name())

    def test_supports_tags(self):
        self.assertTrue(self.format.supports_tags())