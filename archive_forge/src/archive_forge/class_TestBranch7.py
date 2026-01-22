from io import StringIO
from .. import bedding
from .. import branch as _mod_branch
from .. import config, controldir, errors, tests, trace, urlutils
from ..bzr import branch as _mod_bzrbranch
from ..bzr import bzrdir
from ..bzr.fullhistory import BzrBranch5, BzrBranchFormat5
class TestBranch7(TestBranch67, tests.TestCaseWithTransport):

    def get_class(self):
        return _mod_bzrbranch.BzrBranch7

    def get_format_name(self):
        return '1.9'

    def get_format_name_subtree(self):
        return 'development-subtree'

    def test_set_stacked_on_url_unstackable_repo(self):
        repo = self.make_repository('a', format='dirstate-tags')
        control = repo.controldir
        branch = _mod_bzrbranch.BzrBranchFormat7().initialize(control)
        target = self.make_branch('b')
        self.assertRaises(errors.UnstackableRepositoryFormat, branch.set_stacked_on_url, target.base)

    def test_clone_stacked_on_unstackable_repo(self):
        repo = self.make_repository('a', format='dirstate-tags')
        control = repo.controldir
        branch = _mod_bzrbranch.BzrBranchFormat7().initialize(control)
        cloned_bzrdir = control.clone('cloned')

    def _test_default_stacked_location(self):
        branch = self.make_branch('a', format=self.get_format_name())
        self.assertRaises(errors.NotStacked, branch.get_stacked_on_url)

    def test_stack_and_unstack(self):
        branch = self.make_branch('a', format=self.get_format_name())
        target = self.make_branch_and_tree('b', format=self.get_format_name())
        branch.set_stacked_on_url(target.branch.base)
        self.assertEqual(target.branch.base, branch.get_stacked_on_url())
        revid = target.commit('foo')
        self.assertTrue(branch.repository.has_revision(revid))
        branch.set_stacked_on_url(None)
        self.assertRaises(errors.NotStacked, branch.get_stacked_on_url)
        self.assertFalse(branch.repository.has_revision(revid))

    def test_open_opens_stacked_reference(self):
        branch = self.make_branch('a', format=self.get_format_name())
        target = self.make_branch_and_tree('b', format=self.get_format_name())
        branch.set_stacked_on_url(target.branch.base)
        branch = branch.controldir.open_branch()
        revid = target.commit('foo')
        self.assertTrue(branch.repository.has_revision(revid))