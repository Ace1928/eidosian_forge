import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_only_pushes_history(self):
    format = bzrdir.BzrDirMetaFormat1()
    format.repository_format = knitrepo.RepositoryFormatKnit1()
    shared_repo = self.make_repository('repo', format=format, shared=True)
    shared_repo.set_make_working_trees(True)

    def make_shared_tree(path):
        shared_repo.controldir.root_transport.mkdir(path)
        controldir.ControlDir.create_branch_convenience('repo/' + path)
        return workingtree.WorkingTree.open('repo/' + path)
    tree_a = make_shared_tree('a')
    self.build_tree(['repo/a/file'])
    tree_a.add('file')
    tree_a.commit('commit a-1', rev_id=b'a-1')
    f = open('repo/a/file', 'ab')
    f.write(b'more stuff\n')
    f.close()
    tree_a.commit('commit a-2', rev_id=b'a-2')
    tree_b = make_shared_tree('b')
    self.build_tree(['repo/b/file'])
    tree_b.add('file')
    tree_b.commit('commit b-1', rev_id=b'b-1')
    self.assertTrue(shared_repo.has_revision(b'a-1'))
    self.assertTrue(shared_repo.has_revision(b'a-2'))
    self.assertTrue(shared_repo.has_revision(b'b-1'))
    self.run_bzr('push ../../push-b', working_dir='repo/b')
    pushed_tree = workingtree.WorkingTree.open('push-b')
    pushed_repo = pushed_tree.branch.repository
    self.assertFalse(pushed_repo.has_revision(b'a-1'))
    self.assertFalse(pushed_repo.has_revision(b'a-2'))
    self.assertTrue(pushed_repo.has_revision(b'b-1'))