from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def test_simple_bzr_to_git_revision_id(self):
    self.assertEqual((b'c6a4d8f1fa4ac650748e647c4b1b368f589a7356', BzrGitMappingv1()), BzrGitMappingv1().revision_id_bzr_to_foreign(b'git-v1:c6a4d8f1fa4ac650748e647c4b1b368f589a7356'))