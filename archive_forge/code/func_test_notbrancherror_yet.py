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
def test_notbrancherror_yet(self):
    self.assertEqual(NotBranchError('http://', 'A repository for this project does not exist yet.'), parse_git_hangup('http://', HangupException([b'=======', b'', b'A repository for this project does not exist yet.', b'', b'======'])))