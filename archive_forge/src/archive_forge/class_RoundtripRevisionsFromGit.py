from dulwich.objects import Blob, Commit, Tag, parse_timezone
from dulwich.tests.utils import make_object
from ...revision import Revision
from .. import tests
from ..mapping import (BzrGitMappingv1, UnknownCommitEncoding,
class RoundtripRevisionsFromGit(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.mapping = BzrGitMappingv1()

    def assertRoundtripTree(self, tree):
        raise NotImplementedError(self.assertRoundtripTree)

    def assertRoundtripBlob(self, blob):
        raise NotImplementedError(self.assertRoundtripBlob)

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

    def test_commit(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer <Committer>'
        c.commit_time = 4
        c.commit_timezone = -60 * 3
        c.author_time = 5
        c.author_timezone = 60 * 2
        c.author = b'Author <author>'
        self.assertRoundtripCommit(c)

    def test_commit_double_negative_timezone(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer <Committer>'
        c.commit_time = 4
        c.commit_timezone, c._commit_timezone_neg_utc = parse_timezone(b'--700')
        c.author_time = 5
        c.author_timezone = 60 * 2
        c.author = b'Author <author>'
        self.assertRoundtripCommit(c)

    def test_commit_zero_utc_timezone(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer <Committer>'
        c.commit_time = 4
        c.commit_timezone = 0
        c._commit_timezone_neg_utc = True
        c.author_time = 5
        c.author_timezone = 60 * 2
        c.author = b'Author <author>'
        self.assertRoundtripCommit(c)

    def test_commit_encoding(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer <Committer>'
        c.encoding = b'iso8859-1'
        c.commit_time = 4
        c.commit_timezone = -60 * 3
        c.author_time = 5
        c.author_timezone = 60 * 2
        c.author = b'Author <author>'
        self.assertRoundtripCommit(c)

    def test_commit_extra(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer <Committer>'
        c.commit_time = 4
        c.commit_timezone = -60 * 3
        c.author_time = 5
        c.author_timezone = 60 * 2
        c.author = b'Author <author>'
        c._extra = [(b'HG:rename-source', b'hg')]
        self.assertRoundtripCommit(c)

    def test_commit_mergetag(self):
        c = Commit()
        c.tree = b'cc9462f7f8263ef5adfbeff2fb936bb36b504cba'
        c.message = b'Some message'
        c.committer = b'Committer <Committer>'
        c.commit_time = 4
        c.commit_timezone = -60 * 3
        c.author_time = 5
        c.author_timezone = 60 * 2
        c.author = b'Author <author>'
        tag = make_object(Tag, tagger=b'Jelmer Vernooij <jelmer@samba.org>', name=b'0.1', message=None, object=(Blob, b'd80c186a03f423a81b39df39dc87fd269736ca86'), tag_time=423423423, tag_timezone=0)
        c.mergetag = [tag]
        self.assertRoundtripCommit(c)