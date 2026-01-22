import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
class TestUpdate(tests.TestCaseWithTransport):

    def test_update_standalone_trivial(self):
        self.make_branch_and_tree('.')
        out, err = self.run_bzr('update')
        self.assertEqual('Tree is up to date at revision 0 of branch %s\n' % self.test_dir, err)
        self.assertEqual('', out)

    def test_update_quiet(self):
        self.make_branch_and_tree('.')
        out, err = self.run_bzr('update --quiet')
        self.assertEqual('', err)
        self.assertEqual('', out)

    def test_update_standalone_trivial_with_alias_up(self):
        self.make_branch_and_tree('.')
        out, err = self.run_bzr('up')
        self.assertEqual('Tree is up to date at revision 0 of branch %s\n' % self.test_dir, err)
        self.assertEqual('', out)

    def test_update_up_to_date_light_checkout(self):
        self.make_branch_and_tree('branch')
        self.run_bzr('checkout --lightweight branch checkout')
        out, err = self.run_bzr('update checkout')
        self.assertEqual('Tree is up to date at revision 0 of branch %s\n' % osutils.pathjoin(self.test_dir, 'branch'), err)
        self.assertEqual('', out)

    def test_update_up_to_date_checkout(self):
        self.make_branch_and_tree('branch')
        self.run_bzr('checkout branch checkout')
        sr = ScriptRunner()
        sr.run_script(self, '\n$ brz update checkout\n2>Tree is up to date at revision 0 of branch .../branch\n')

    def test_update_out_of_date_standalone_tree(self):
        self.make_branch_and_tree('branch')
        self.run_bzr('checkout --lightweight branch checkout')
        self.build_tree(['checkout/file'])
        self.run_bzr('add checkout/file')
        self.run_bzr('commit -m add-file checkout')
        out, err = self.run_bzr('update branch')
        self.assertEqual('', out)
        self.assertEqualDiff('+N  file\nAll changes applied successfully.\nUpdated to revision 1 of branch %s\n' % osutils.pathjoin(self.test_dir, 'branch'), err)
        self.assertPathExists('branch/file')

    def test_update_out_of_date_light_checkout(self):
        self.make_branch_and_tree('branch')
        self.run_bzr('checkout --lightweight branch checkout')
        self.run_bzr('checkout --lightweight branch checkout2')
        self.build_tree(['checkout/file'])
        self.run_bzr('add checkout/file')
        self.run_bzr('commit -m add-file checkout')
        out, err = self.run_bzr('update checkout2')
        self.assertEqualDiff('+N  file\nAll changes applied successfully.\nUpdated to revision 1 of branch %s\n' % osutils.pathjoin(self.test_dir, 'branch'), err)
        self.assertEqual('', out)

    def test_update_conflicts_returns_2(self):
        self.make_branch_and_tree('branch')
        self.run_bzr('checkout --lightweight branch checkout')
        self.build_tree(['checkout/file'])
        self.run_bzr('add checkout/file')
        self.run_bzr('commit -m add-file checkout')
        self.run_bzr('checkout --lightweight branch checkout2')
        with open('checkout/file', 'w') as a_file:
            a_file.write('Foo')
        self.run_bzr('commit -m checnge-file checkout')
        with open('checkout2/file', 'w') as a_file:
            a_file.write('Bar')
        out, err = self.run_bzr('update checkout2', retcode=1)
        self.assertEqualDiff(' M  file\nText conflict in file\n1 conflicts encountered.\nUpdated to revision 2 of branch %s\n' % osutils.pathjoin(self.test_dir, 'branch'), err)
        self.assertEqual('', out)

    def test_smoke_update_checkout_bound_branch_local_commits(self):
        master = self.make_branch_and_tree('master')
        master.commit('first commit')
        self.run_bzr('checkout master child')
        self.run_bzr('checkout --lightweight child checkout')
        wt = workingtree.WorkingTree.open('checkout')
        with open('master/file', 'w') as a_file:
            a_file.write('Foo')
        master.add(['file'])
        master_tip = master.commit('add file')
        with open('child/file_b', 'w') as a_file:
            a_file.write('Foo')
        child = workingtree.WorkingTree.open('child')
        child.add(['file_b'])
        child_tip = child.commit('add file_b', local=True)
        with open('checkout/file_c', 'w') as a_file:
            a_file.write('Foo')
        wt.add(['file_c'])
        out, err = self.run_bzr('update checkout')
        self.assertEqual('', out)
        self.assertEqualDiff("+N  file_b\nAll changes applied successfully.\n+N  file\nAll changes applied successfully.\nUpdated to revision 2 of branch %s\nYour local commits will now show as pending merges with 'brz status', and can be committed with 'brz commit'.\n" % osutils.pathjoin(self.test_dir, 'master'), err)
        self.assertEqual([master_tip, child_tip], wt.get_parent_ids())
        self.assertPathExists('checkout/file')
        self.assertPathExists('checkout/file_b')
        self.assertPathExists('checkout/file_c')
        self.assertTrue(wt.has_filename('file_c'))

    def test_update_with_merges(self):
        master = self.make_branch_and_tree('master')
        self.build_tree(['master/file'])
        master.add(['file'])
        master.commit('one', rev_id=b'm1')
        self.build_tree(['checkout1/'])
        checkout_dir = bzrdir.BzrDirMetaFormat1().initialize('checkout1')
        checkout_dir.set_branch_reference(master.branch)
        checkout1 = checkout_dir.create_workingtree(b'm1')
        other = master.controldir.sprout('other').open_workingtree()
        self.build_tree(['other/file2'])
        other.add(['file2'])
        other.commit('other2', rev_id=b'o2')
        self.build_tree(['master/file3'])
        master.add(['file3'])
        master.commit('f3', rev_id=b'm2')
        os.chdir('checkout1')
        self.run_bzr('merge ../other')
        self.assertEqual([b'o2'], checkout1.get_parent_ids()[1:])
        self.run_bzr_error(["please run 'brz update'"], 'commit -m merged')
        out, err = self.run_bzr('update')
        self.assertEqual('', out)
        self.assertEqualDiff('+N  file3\nAll changes applied successfully.\nUpdated to revision 2 of branch %s\n' % osutils.pathjoin(self.test_dir, 'master'), err)
        self.assertEqual([b'o2'], checkout1.get_parent_ids()[1:])

    def test_readonly_lightweight_update(self):
        """Update a light checkout of a readonly branch"""
        tree = self.make_branch_and_tree('branch')
        readonly_branch = branch.Branch.open(self.get_readonly_url('branch'))
        checkout = readonly_branch.create_checkout('checkout', lightweight=True)
        tree.commit('empty commit')
        self.run_bzr('update checkout')

    def test_update_with_merge_merged_to_master(self):
        master = self.make_branch_and_tree('master')
        self.build_tree(['master/file'])
        master.add(['file'])
        master.commit('one', rev_id=b'm1')
        self.build_tree(['checkout1/'])
        checkout_dir = bzrdir.BzrDirMetaFormat1().initialize('checkout1')
        checkout_dir.set_branch_reference(master.branch)
        checkout1 = checkout_dir.create_workingtree(b'm1')
        other = master.controldir.sprout('other').open_workingtree()
        self.build_tree(['other/file2'])
        other.add(['file2'])
        other.commit('other2', rev_id=b'o2')
        checkout1.merge_from_branch(other.branch)
        self.assertEqual([b'o2'], checkout1.get_parent_ids()[1:])
        master.merge_from_branch(other.branch)
        master.commit('f3', rev_id=b'm2')
        out, err = self.run_bzr('update', working_dir='checkout1')
        self.assertEqual('', out)
        self.assertEqualDiff('All changes applied successfully.\nUpdated to revision 2 of branch %s\n' % osutils.pathjoin(self.test_dir, 'master'), err)
        self.assertEqual([], checkout1.get_parent_ids()[1:])

    def test_update_dash_r(self):
        master = self.make_branch_and_tree('master')
        os.chdir('master')
        self.build_tree(['./file1'])
        master.add(['file1'])
        master.commit('one', rev_id=b'm1')
        self.build_tree(['./file2'])
        master.add(['file2'])
        master.commit('two', rev_id=b'm2')
        sr = ScriptRunner()
        sr.run_script(self, '\n$ brz update -r 1\n2>-D  file2\n2>All changes applied successfully.\n2>Updated to revision 1 of .../master\n')
        self.assertPathExists('./file1')
        self.assertPathDoesNotExist('./file2')
        self.assertEqual([b'm1'], master.get_parent_ids())

    def test_update_dash_r_outside_history(self):
        """Ensure that we can update -r to dotted revisions.
        """
        master = self.make_branch_and_tree('master')
        self.build_tree(['master/file1'])
        master.add(['file1'])
        master.commit('one', rev_id=b'm1')
        other = master.controldir.sprout('other').open_workingtree()
        self.build_tree(['other/file2', 'other/file3'])
        other.add(['file2'])
        other.commit('other2', rev_id=b'o2')
        other.add(['file3'])
        other.commit('other3', rev_id=b'o3')
        os.chdir('master')
        self.run_bzr('merge ../other')
        master.commit('merge', rev_id=b'merge')
        out, err = self.run_bzr('update -r revid:o2')
        self.assertContainsRe(err, '-D\\s+file3')
        self.assertContainsRe(err, 'All changes applied successfully\\.')
        self.assertContainsRe(err, 'Updated to revision 1.1.1 of branch .*')
        out, err = self.run_bzr('update')
        self.assertContainsRe(err, '\\+N\\s+file3')
        self.assertContainsRe(err, 'All changes applied successfully\\.')
        self.assertContainsRe(err, 'Updated to revision 2 of branch .*')

    def test_update_dash_r_in_master(self):
        master = self.make_branch_and_tree('master')
        self.build_tree(['master/file1'])
        master.add(['file1'])
        master.commit('one', rev_id=b'm1')
        self.run_bzr('checkout master checkout')
        self.build_tree(['master/file2'])
        master.add(['file2'])
        master.commit('two', rev_id=b'm2')
        os.chdir('checkout')
        sr = ScriptRunner()
        sr.run_script(self, '\n$ brz update -r revid:m2\n2>+N  file2\n2>All changes applied successfully.\n2>Updated to revision 2 of branch .../master\n')

    def test_update_show_base(self):
        """brz update support --show-base

        see https://bugs.launchpad.net/bzr/+bug/202374"""
        tree = self.make_branch_and_tree('.')
        with open('hello', 'w') as f:
            f.write('foo')
        tree.add('hello')
        tree.commit('fie')
        with open('hello', 'w') as f:
            f.write('fee')
        tree.commit('fee')
        self.run_bzr(['update', '-r1'])
        with open('hello', 'w') as f:
            f.write('fie')
        out, err = self.run_bzr(['update', '--show-base'], retcode=1)
        self.assertContainsString(err, ' M  hello\nText conflict in hello\n1 conflicts encountered.\n')
        with open('hello', 'rb') as f:
            self.assertEqualDiff(b'<<<<<<< TREE\nfie||||||| BASE-REVISION\nfoo=======\nfee>>>>>>> MERGE-SOURCE\n', f.read())

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

    def test_no_upgrade_single_file(self):
        """There's one basis revision per tree.

        Since you can't actually change the basis for a single file at the
        moment, we don't let you think you can.

        See bug 557886.
        """
        self.make_branch_and_tree('.')
        self.build_tree_contents([('a/',), ('a/file', b'content')])
        sr = ScriptRunner()
        sr.run_script(self, '\n            $ brz update ./a\n            2>brz: ERROR: brz update can only update a whole tree, not a file or subdirectory\n            $ brz update ./a/file\n            2>brz: ERROR: brz update can only update a whole tree, not a file or subdirectory\n            $ brz update .\n            2>Tree is up to date at revision 0 of branch ...\n            $ cd a\n            $ brz update .\n            2>brz: ERROR: brz update can only update a whole tree, not a file or subdirectory\n            # however, you can update the whole tree from a subdirectory\n            $ brz update\n            2>Tree is up to date at revision 0 of branch ...\n            ')