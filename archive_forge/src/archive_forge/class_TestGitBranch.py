import os
import dulwich
from dulwich.objects import Commit, Tag
from dulwich.repo import Repo as GitRepo
from ... import errors, revision, urlutils
from ...branch import Branch, InterBranch, UnstackableBranchFormat
from ...controldir import ControlDir
from ...repository import Repository
from .. import branch, tests
from ..dir import LocalGitControlDirFormat
from ..mapping import default_mapping
class TestGitBranch(tests.TestCaseInTempDir):

    def test_open_by_ref(self):
        GitRepo.init('.')
        url = '{},ref={}'.format(urlutils.local_path_to_url(self.test_dir), urlutils.quote('refs/remotes/origin/unstable', safe=''))
        d = ControlDir.open(url)
        b = d.create_branch()
        self.assertEqual(b.ref, b'refs/remotes/origin/unstable')

    def test_open_existing(self):
        r = GitRepo.init('.')
        d = ControlDir.open('.')
        thebranch = d.create_branch()
        self.assertIsInstance(thebranch, branch.GitBranch)

    def test_repr(self):
        r = GitRepo.init('.')
        d = ControlDir.open('.')
        thebranch = d.create_branch()
        self.assertEqual("<LocalGitBranch('{}/', {!r})>".format(urlutils.local_path_to_url(self.test_dir), 'master'), repr(thebranch))

    def test_last_revision_is_null(self):
        r = GitRepo.init('.')
        thedir = ControlDir.open('.')
        thebranch = thedir.create_branch()
        self.assertEqual(revision.NULL_REVISION, thebranch.last_revision())
        self.assertEqual((0, revision.NULL_REVISION), thebranch.last_revision_info())

    def simple_commit_a(self):
        r = GitRepo.init('.')
        self.build_tree(['a'])
        r.stage(['a'])
        return r.do_commit(b'a', committer=b'Somebody <foo@example.com>')

    def test_last_revision_is_valid(self):
        head = self.simple_commit_a()
        thebranch = Branch.open('.')
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(head), thebranch.last_revision())

    def test_last_revision_info(self):
        reva = self.simple_commit_a()
        self.build_tree(['b'])
        r = GitRepo('.')
        self.addCleanup(r.close)
        r.stage('b')
        revb = r.do_commit(b'b', committer=b'Somebody <foo@example.com>')
        thebranch = Branch.open('.')
        self.assertEqual((2, default_mapping.revision_id_foreign_to_bzr(revb)), thebranch.last_revision_info())

    def test_tag_annotated(self):
        reva = self.simple_commit_a()
        o = Tag()
        o.name = b'foo'
        o.tagger = b'Jelmer <foo@example.com>'
        o.message = b'add tag'
        o.object = (Commit, reva)
        o.tag_timezone = 0
        o.tag_time = 42
        r = GitRepo('.')
        self.addCleanup(r.close)
        r.object_store.add_object(o)
        r[b'refs/tags/foo'] = o.id
        thebranch = Branch.open('.')
        self.assertEqual({'foo': default_mapping.revision_id_foreign_to_bzr(reva)}, thebranch.tags.get_tag_dict())

    def test_tag(self):
        reva = self.simple_commit_a()
        r = GitRepo('.')
        self.addCleanup(r.close)
        r.refs[b'refs/tags/foo'] = reva
        thebranch = Branch.open('.')
        self.assertEqual({'foo': default_mapping.revision_id_foreign_to_bzr(reva)}, thebranch.tags.get_tag_dict())