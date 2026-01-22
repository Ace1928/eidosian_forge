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
class FetchFromRemoteTestBase:
    _test_needs_features = [ExecutableFeature('git')]
    _to_format: str

    def setUp(self):
        TestCaseWithTransport.setUp(self)
        self.remote_real = GitRepo.init('remote', mkdir=True)
        self.remote_url = 'git://%s/' % os.path.abspath(self.remote_real.path)
        self.permit_url(self.remote_url)

    def test_sprout_simple(self):
        self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        remote = ControlDir.open(self.remote_url)
        self.make_controldir('local', format=self._to_format)
        local = remote.sprout('local')
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(self.remote_real.head()), local.open_branch().last_revision())

    def test_sprout_submodule_invalid(self):
        self.sub_real = GitRepo.init('sub', mkdir=True)
        self.sub_real.do_commit(message=b'message in sub', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        self.sub_real.clone('remote/nested')
        self.remote_real.stage('nested')
        self.permit_url(urljoin(self.remote_url, '../sub'))
        self.assertIn(b'nested', self.remote_real.open_index())
        self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        remote = ControlDir.open(self.remote_url)
        self.make_controldir('local', format=self._to_format)
        local = remote.sprout('local')
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(self.remote_real.head()), local.open_branch().last_revision())
        self.assertRaises(MissingNestedTree, local.open_workingtree().get_nested_tree, 'nested')

    def test_sprout_submodule_relative(self):
        self.sub_real = GitRepo.init('sub', mkdir=True)
        self.sub_real.do_commit(message=b'message in sub', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        with open('remote/.gitmodules', 'w') as f:
            f.write('\n[submodule "lala"]\n\tpath = nested\n\turl = ../sub/.git\n')
        self.remote_real.stage('.gitmodules')
        self.sub_real.clone('remote/nested')
        self.remote_real.stage('nested')
        self.permit_url(urljoin(self.remote_url, '../sub'))
        self.assertIn(b'nested', self.remote_real.open_index())
        self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        remote = ControlDir.open(self.remote_url)
        self.make_controldir('local', format=self._to_format)
        local = remote.sprout('local')
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(self.remote_real.head()), local.open_branch().last_revision())
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(self.sub_real.head()), local.open_workingtree().get_nested_tree('nested').last_revision())

    def test_sprout_with_tags(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/tags/another')
        self.remote_real.refs[b'refs/tags/blah'] = self.remote_real.head()
        remote = ControlDir.open(self.remote_url)
        self.make_controldir('local', format=self._to_format)
        local = remote.sprout('local')
        local_branch = local.open_branch()
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(c1), local_branch.last_revision())
        self.assertEqual({'blah': local_branch.last_revision(), 'another': default_mapping.revision_id_foreign_to_bzr(c2)}, local_branch.tags.get_tag_dict())

    def test_sprout_with_annotated_tag(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>', ref=b'refs/heads/another')
        porcelain.tag_create(self.remote_real, tag=b'blah', author=b'author <author@example.com>', objectish=c2, tag_time=int(time.time()), tag_timezone=0, annotated=True, message=b'Annotated tag')
        remote = ControlDir.open(self.remote_url)
        self.make_controldir('local', format=self._to_format)
        local = remote.sprout('local', revision_id=default_mapping.revision_id_foreign_to_bzr(c1))
        local_branch = local.open_branch()
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(c1), local_branch.last_revision())
        self.assertEqual({'blah': default_mapping.revision_id_foreign_to_bzr(c2)}, local_branch.tags.get_tag_dict())

    def test_sprout_with_annotated_tag_unreferenced(self):
        c1 = self.remote_real.do_commit(message=b'message', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        c2 = self.remote_real.do_commit(message=b'another commit', committer=b'committer <committer@example.com>', author=b'author <author@example.com>')
        porcelain.tag_create(self.remote_real, tag=b'blah', author=b'author <author@example.com>', objectish=c1, tag_time=int(time.time()), tag_timezone=0, annotated=True, message=b'Annotated tag')
        remote = ControlDir.open(self.remote_url)
        self.make_controldir('local', format=self._to_format)
        local = remote.sprout('local', revision_id=default_mapping.revision_id_foreign_to_bzr(c1))
        local_branch = local.open_branch()
        self.assertEqual(default_mapping.revision_id_foreign_to_bzr(c1), local_branch.last_revision())
        self.assertEqual({'blah': default_mapping.revision_id_foreign_to_bzr(c1)}, local_branch.tags.get_tag_dict())