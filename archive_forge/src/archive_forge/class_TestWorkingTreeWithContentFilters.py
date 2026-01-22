import os
from breezy.controldir import ControlDir
from breezy.filters import ContentFilter
from breezy.switch import switch
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.workingtree import WorkingTree
class TestWorkingTreeWithContentFilters(TestCaseWithWorkingTree):

    def create_cf_tree(self, txt_reader, txt_writer, dir='.'):
        tree = self.make_branch_and_tree(dir)

        def _content_filter_stack(path=None, file_id=None):
            if path.endswith('.txt'):
                return [ContentFilter(txt_reader, txt_writer)]
            else:
                return []
        tree._content_filter_stack = _content_filter_stack
        self.build_tree_contents([(dir + '/file1.txt', b'Foo Txt'), (dir + '/file2.bin', b'Foo Bin')])
        tree.add(['file1.txt', 'file2.bin'])
        tree.commit('commit raw content')
        return (tree, 'file1.txt', 'file2.bin')

    def create_cf_tree_with_two_revisions(self, txt_reader, txt_writer, dir='.'):
        tree = self.make_branch_and_tree(dir)

        def _content_filter_stack(path=None, file_id=None):
            if path.endswith('.txt'):
                return [ContentFilter(txt_reader, txt_writer)]
            else:
                return []
        tree._content_filter_stack = _content_filter_stack
        self.build_tree_contents([(dir + '/file1.txt', b'Foo Txt'), (dir + '/file2.bin', b'Foo Bin'), (dir + '/file3.txt', b'Bar Txt')])
        tree.add(['file1.txt', 'file2.bin', 'file3.txt'])
        tree.commit('commit raw content')
        self.build_tree_contents([(dir + '/file1.txt', b'Foo ROCKS!'), (dir + '/file4.txt', b'Hello World')])
        tree.add(['file4.txt'])
        tree.remove(['file3.txt'], keep_files=False)
        tree.commit('change, add and rename stuff')
        return (tree, 'file1.txt', 'file2.bin', 'file3.txt', 'file4.txt')

    def patch_in_content_filter(self):

        def new_stack(tree, path=None, file_id=None):
            if path.endswith('.txt'):
                return [ContentFilter(_swapcase, _swapcase)]
            else:
                return []
        self.overrideAttr(WorkingTree, '_content_filter_stack', new_stack)

    def assert_basis_content(self, expected_content, branch, path):
        basis = branch.basis_tree()
        with basis.lock_read():
            self.assertEqual(expected_content, basis.get_file_text(path))

    def test_symmetric_content_filtering(self):
        tree, txt_path, bin_path = self.create_cf_tree(txt_reader=_swapcase, txt_writer=_swapcase)
        basis = tree.basis_tree()
        basis.lock_read()
        self.addCleanup(basis.unlock)
        if tree.supports_content_filtering():
            expected = b'fOO tXT'
        else:
            expected = b'Foo Txt'
        self.assertEqual(expected, basis.get_file_text(txt_path))
        self.assertEqual(b'Foo Bin', basis.get_file_text(bin_path))
        tree.lock_read()
        self.addCleanup(tree.unlock)
        with tree.get_file(txt_path, filtered=False) as f:
            self.assertEqual(b'Foo Txt', f.read())
        with tree.get_file(bin_path, filtered=False) as f:
            self.assertEqual(b'Foo Bin', f.read())

    def test_readonly_content_filtering(self):
        tree, txt_path, bin_path = self.create_cf_tree(txt_reader=_uppercase, txt_writer=None)
        basis = tree.basis_tree()
        basis.lock_read()
        self.addCleanup(basis.unlock)
        if tree.supports_content_filtering():
            expected = b'FOO TXT'
        else:
            expected = b'Foo Txt'
        self.assertEqual(expected, basis.get_file_text(txt_path))
        self.assertEqual(b'Foo Bin', basis.get_file_text(bin_path))
        tree.lock_read()
        self.addCleanup(tree.unlock)
        with tree.get_file(txt_path, filtered=False) as f:
            self.assertEqual(b'Foo Txt', f.read())
        with tree.get_file(bin_path, filtered=False) as f:
            self.assertEqual(b'Foo Bin', f.read())

    def test_branch_source_filtered_target_not(self):
        source, txt_path, bin_path = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual(b'Foo Txt', 'source/file1.txt')
        self.assert_basis_content(b'FOO TXT', source, txt_path)
        self.run_bzr('branch source target')
        target = WorkingTree.open('target')
        self.assertFileEqual(b'FOO TXT', 'target/file1.txt')
        changes = target.changes_from(source.basis_tree())
        self.assertFalse(changes.has_changed())

    def test_branch_source_not_filtered_target_is(self):
        source, txt_path, bin_path = self.create_cf_tree(txt_reader=None, txt_writer=None, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual(b'Foo Txt', 'source/file1.txt')
        self.assert_basis_content(b'Foo Txt', source, txt_path)
        self.patch_in_content_filter()
        self.run_bzr('branch source target')
        target = WorkingTree.open('target')
        self.assertFileEqual(b'fOO tXT', 'target/file1.txt')
        changes = target.changes_from(source.basis_tree())
        self.assertFalse(changes.has_changed())

    def test_path_content_summary(self):
        """path_content_summary should always talk about the canonical form."""
        source, txt_path, bin_path = self.create_cf_tree(txt_reader=_append_text, txt_writer=_remove_appended_text, dir='source')
        if not source.supports_content_filtering():
            return
        source.lock_read()
        self.addCleanup(source.unlock)
        expected_canonical_form = b'Foo Txt\nend string\n'
        with source.get_file(txt_path, filtered=True) as f:
            self.assertEqual(f.read(), expected_canonical_form)
        with source.get_file(txt_path, filtered=False) as f:
            self.assertEqual(f.read(), b'Foo Txt')
        result = source.path_content_summary('file1.txt')
        self.assertEqual(result, ('file', None, False, None))

    def test_content_filtering_applied_on_pull(self):
        source, path1, path2, path3, path4 = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual(b'Foo ROCKS!', 'source/file1.txt')
        self.assert_basis_content(b'Foo ROCKS!', source, path1)
        self.patch_in_content_filter()
        self.run_bzr('branch -r1 source target')
        target = WorkingTree.open('target')
        self.assert_basis_content(b'Foo Txt', target, path1)
        self.assertFileEqual(b'fOO tXT', 'target/file1.txt')
        self.assert_basis_content(b'Foo Bin', target, path2)
        self.assertFileEqual(b'Foo Bin', 'target/file2.bin')
        self.assert_basis_content(b'Bar Txt', target, path3)
        self.assertFileEqual(b'bAR tXT', 'target/file3.txt')
        self.run_bzr('pull -d target')
        self.assert_basis_content(b'Foo ROCKS!', target, path1)
        self.assertFileEqual(b'fOO rocks!', 'target/file1.txt')
        self.assert_basis_content(b'Foo Bin', target, path2)
        self.assert_basis_content(b'Hello World', target, path4)
        self.assertFileEqual(b'hELLO wORLD', 'target/file4.txt')

    def test_content_filtering_applied_on_merge(self):
        source, path1, path2, path3, path4 = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='source')
        if not source.supports_content_filtering():
            return
        self.assert_basis_content(b'Foo ROCKS!', source, path1)
        self.assertFileEqual(b'Foo ROCKS!', 'source/file1.txt')
        self.assert_basis_content(b'Foo Bin', source, path2)
        self.assert_basis_content(b'Hello World', source, path4)
        self.assertFileEqual(b'Hello World', 'source/file4.txt')
        self.patch_in_content_filter()
        self.run_bzr('branch -r1 source target')
        target = WorkingTree.open('target')
        self.assert_basis_content(b'Foo Txt', target, path1)
        self.assertFileEqual(b'fOO tXT', 'target/file1.txt')
        self.assertFileEqual(b'Foo Bin', 'target/file2.bin')
        self.assertFileEqual(b'bAR tXT', 'target/file3.txt')
        self.run_bzr('merge -d target source')
        self.assertFileEqual(b'fOO rocks!', 'target/file1.txt')
        self.assertFileEqual(b'hELLO wORLD', 'target/file4.txt')
        target.commit('merge file1.txt changes from source')
        self.assert_basis_content(b'Foo ROCKS!', target, path1)
        self.assert_basis_content(b'Hello World', target, path4)

    def test_content_filtering_applied_on_switch(self):
        source, path1, path2, path3, path4 = self.create_cf_tree_with_two_revisions(txt_reader=None, txt_writer=None, dir='branch-a')
        if not source.supports_content_filtering():
            return
        self.patch_in_content_filter()
        self.run_bzr('branch -r1 branch-a branch-b')
        self.run_bzr('checkout --lightweight branch-b checkout')
        self.assertFileEqual(b'fOO tXT', 'checkout/file1.txt')
        checkout_control_dir = ControlDir.open_containing('checkout')[0]
        switch(checkout_control_dir, source.branch)
        self.assertFileEqual(b'fOO rocks!', 'checkout/file1.txt')
        self.assertFileEqual(b'hELLO wORLD', 'checkout/file4.txt')

    def test_content_filtering_applied_on_revert_delete(self):
        source, txt_path, bin_path = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual(b'Foo Txt', 'source/file1.txt')
        self.assert_basis_content(b'FOO TXT', source, txt_path)
        os.unlink('source/file1.txt')
        self.assertFalse(os.path.exists('source/file1.txt'))
        source.revert(['file1.txt'])
        self.assertTrue(os.path.exists('source/file1.txt'))
        self.assertFileEqual(b'foo txt', 'source/file1.txt')

    def test_content_filtering_applied_on_revert_rename(self):
        source, txt_path, bin_path = self.create_cf_tree(txt_reader=_uppercase, txt_writer=_lowercase, dir='source')
        if not source.supports_content_filtering():
            return
        self.assertFileEqual(b'Foo Txt', 'source/file1.txt')
        self.assert_basis_content(b'FOO TXT', source, txt_path)
        self.build_tree_contents([('source/file1.txt', b'Foo Txt with new content')])
        source.rename_one('file1.txt', 'file1.bin')
        self.assertTrue(os.path.exists('source/file1.bin'))
        self.assertFalse(os.path.exists('source/file1.txt'))
        self.assertFileEqual(b'Foo Txt with new content', 'source/file1.bin')
        source.revert(['file1.bin'])
        self.assertFalse(os.path.exists('source/file1.bin'))
        self.assertTrue(os.path.exists('source/file1.txt'))
        self.assertFileEqual(b'foo txt', 'source/file1.txt')