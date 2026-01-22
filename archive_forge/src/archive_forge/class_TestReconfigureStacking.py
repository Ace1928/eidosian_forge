from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
class TestReconfigureStacking(tests.TestCaseWithTransport):

    def test_reconfigure_stacking(self):
        """Test a fairly realistic scenario for stacking:

         * make a branch with some history
         * branch it
         * make the second branch stacked on the first
         * commit in the second
         * then make the second unstacked, so it has to fill in history from
           the original fallback lying underneath its original content

        See discussion in <https://bugs.launchpad.net/bzr/+bug/391411>
        """
        tree_1 = self.make_branch_and_tree('b1', format='2a')
        self.build_tree(['b1/foo'])
        tree_1.add(['foo'])
        tree_1.commit('add foo')
        branch_1 = tree_1.branch
        bzrdir_2 = tree_1.controldir.sprout('b2')
        tree_2 = bzrdir_2.open_workingtree()
        branch_2 = tree_2.branch
        out, err = self.run_bzr('reconfigure --stacked-on b1 b2')
        self.assertContainsRe(out, '^.*/b2/ is now stacked on ../b1\n$')
        self.assertEqual('', err)
        out, err = self.run_bzr('reconfigure --stacked-on %s b2' % (self.get_url('b1'),))
        self.assertContainsRe(out, '^.*/b2/ is now stacked on ../b1\n$')
        self.assertEqual('', err)
        branch_2 = branch_2.controldir.open_branch()
        self.assertEqual('../b1', branch_2.get_stacked_on_url())
        self.build_tree_contents([('foo', b'new foo')])
        tree_2.commit('update foo')
        out, err = self.run_bzr('reconfigure --unstacked b2')
        self.assertContainsRe(out, '^.*/b2/ is now not stacked\n$')
        self.assertEqual('', err)
        branch_2 = branch_2.controldir.open_branch()
        self.assertRaises(errors.NotStacked, branch_2.get_stacked_on_url)