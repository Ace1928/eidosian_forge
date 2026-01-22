from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
def assertRoundtripCommit(self, commit1):
    rev, roundtrip_revid, verifiers = self.mapping.import_commit(commit1, self.mapping.revision_id_foreign_to_bzr, strict=True)
    commit2 = self.mapping.export_commit(rev, '12341212121212', None, True, None)
    self.assertEqual(commit1.committer, commit2.committer)
    self.assertEqual(commit1.commit_time, commit2.commit_time)
    self.assertEqual(commit1.commit_timezone, commit2.commit_timezone)
    self.assertEqual(commit1.author, commit2.author)
    self.assertEqual(commit1.author_time, commit2.author_time)
    self.assertEqual(commit1.author_timezone, commit2.author_timezone)
    self.assertEqual(commit1.message, commit2.message)
    self.assertEqual(commit1.encoding, commit2.encoding)