import re
from io import BytesIO
from ... import branch as _mod_branch
from ... import commit, controldir
from ... import delta as _mod_delta
from ... import errors, gpg, info, repository
from ... import revision as _mod_revision
from ... import tests, transport, upgrade, workingtree
from ...bzr import branch as _mod_bzrbranch
from ...bzr import inventory, knitpack_repo, remote
from ...bzr import repository as bzrrepository
from .. import per_repository, test_server
from ..matchers import *
class TestDeltaRevisionFilesFiltered(per_repository.TestCaseWithRepository):

    def setUp(self):
        super().setUp()
        self.tree_a = self.make_branch_and_tree('a')
        self.build_tree(['a/foo', 'a/bar/', 'a/bar/b1', 'a/bar/b2', 'a/baz', 'a/oldname'])
        self.tree_a.add(['foo', 'bar', 'bar/b1', 'bar/b2', 'baz', 'oldname'])
        self.rev1 = self.tree_a.commit('rev1')
        self.build_tree(['a/bar/b3'])
        self.tree_a.add('bar/b3')
        self.tree_a.rename_one('oldname', 'newname')
        self.rev2 = self.tree_a.commit('rev2')
        self.repository = self.tree_a.branch.repository
        self.addCleanup(self.repository.lock_read().unlock)

    def test_multiple_files(self):
        delta = list(self.repository.get_revision_deltas([self.repository.get_revision(self.rev1)], specific_files=['foo', 'baz']))[0]
        self.assertIsInstance(delta, _mod_delta.TreeDelta)
        self.assertEqual([('baz', 'file'), ('foo', 'file')], [(c.path[1], c.kind[1]) for c in delta.added])

    def test_directory(self):
        delta = list(self.repository.get_revision_deltas([self.repository.get_revision(self.rev1)], specific_files=['bar']))[0]
        self.assertIsInstance(delta, _mod_delta.TreeDelta)
        self.assertEqual([('bar', 'directory'), ('bar/b1', 'file'), ('bar/b2', 'file')], [(c.path[1], c.kind[1]) for c in delta.added])

    def test_unrelated(self):
        delta = list(self.repository.get_revision_deltas([self.repository.get_revision(self.rev2)], specific_files=['foo']))[0]
        self.assertIsInstance(delta, _mod_delta.TreeDelta)
        self.assertEqual([], delta.added)

    def test_renamed(self):
        self.assertTrue(self.repository.revision_tree(self.rev2).has_filename('newname'))
        self.assertTrue(self.repository.revision_tree(self.rev1).has_filename('oldname'))
        revs = [self.repository.get_revision(self.rev2), self.repository.get_revision(self.rev1)]
        delta2, delta1 = list(self.repository.get_revision_deltas(revs, specific_files=['newname']))
        self.assertIsInstance(delta1, _mod_delta.TreeDelta)
        self.assertEqual([('oldname', 'newname')], [c.path for c in delta2.renamed])
        self.assertIsInstance(delta2, _mod_delta.TreeDelta)
        self.assertEqual(['oldname'], [c.path[1] for c in delta1.added])

    def test_file_in_directory(self):
        delta = list(self.repository.get_revision_deltas([self.repository.get_revision(self.rev1)], specific_files=['bar/b2']))[0]
        self.assertIsInstance(delta, _mod_delta.TreeDelta)
        self.assertEqual([('bar', 'directory'), ('bar/b2', 'file')], [(c.path[1], c.kind[1]) for c in delta.added])

    def test_file_in_unchanged_directory(self):
        delta = list(self.repository.get_revision_deltas([self.repository.get_revision(self.rev2)], specific_files=['bar/b3']))[0]
        self.assertIsInstance(delta, _mod_delta.TreeDelta)
        if [(c.path[1], c.kind[1]) for c in delta.added] == [('bar', 'directory'), ('bar/b3', 'file')]:
            self.knownFailure("bzr incorrectly reports 'bar' as added - bug 878217")
        self.assertEqual([('bar/b3', 'file')], [(c.path[1], c.kind[1]) for c in delta.added])