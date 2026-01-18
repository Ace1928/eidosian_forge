import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_checkout_prevent_double_merge(self):
    """"Launchpad bug 113809 in brz "update performs two merges"
        https://launchpad.net/bugs/113809"""
    master = self.make_branch_and_tree('master')
    self.build_tree_contents([('master/file', b'initial contents\n')])
    master.add(['file'])
    master.commit('one', rev_id=b'm1')
    checkout = master.branch.create_checkout('checkout')
    lightweight = checkout.branch.create_checkout('lightweight', lightweight=True)
    self.build_tree_contents([('master/file', b'master\n')])
    master.commit('two', rev_id=b'm2')
    self.build_tree_contents([('master/file', b'master local changes\n')])
    self.build_tree_contents([('checkout/file', b'checkout\n')])
    checkout.commit('tree', rev_id=b'c2', local=True)
    self.build_tree_contents([('checkout/file', b'checkout local changes\n')])
    self.build_tree_contents([('lightweight/file', b'lightweight local changes\n')])
    out, err = self.run_bzr('update lightweight', retcode=1)
    self.assertEqual('', out)
    self.assertFileEqual('<<<<<<< TREE\nlightweight local changes\n=======\ncheckout\n>>>>>>> MERGE-SOURCE\n', 'lightweight/file')
    self.build_tree_contents([('lightweight/file', b'lightweight+checkout\n')])
    self.run_bzr('resolve lightweight/file')
    out, err = self.run_bzr('update lightweight', retcode=1)
    self.assertEqual('', out)
    self.assertFileEqual('<<<<<<< TREE\nlightweight+checkout\n=======\nmaster\n>>>>>>> MERGE-SOURCE\n', 'lightweight/file')