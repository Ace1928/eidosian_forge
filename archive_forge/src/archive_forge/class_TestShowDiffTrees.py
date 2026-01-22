import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
class TestShowDiffTrees(tests.TestCaseWithTransport):
    """Direct tests for show_diff_trees"""

    def test_modified_file(self):
        """Test when a file is modified."""
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', b'contents\n')])
        tree.add(['file'], ids=[b'file-id'])
        tree.commit('one', rev_id=b'rev-1')
        self.build_tree_contents([('tree/file', b'new contents\n')])
        d = get_diff_as_string(tree.basis_tree(), tree)
        self.assertContainsRe(d, b"=== modified file 'file'\n")
        self.assertContainsRe(d, b'--- old/file\t')
        self.assertContainsRe(d, b'\\+\\+\\+ new/file\t')
        self.assertContainsRe(d, b'-contents\n\\+new contents\n')

    def test_modified_file_in_renamed_dir(self):
        """Test when a file is modified in a renamed directory."""
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/dir/'])
        self.build_tree_contents([('tree/dir/file', b'contents\n')])
        tree.add(['dir', 'dir/file'], ids=[b'dir-id', b'file-id'])
        tree.commit('one', rev_id=b'rev-1')
        tree.rename_one('dir', 'other')
        self.build_tree_contents([('tree/other/file', b'new contents\n')])
        d = get_diff_as_string(tree.basis_tree(), tree)
        self.assertContainsRe(d, b"=== renamed directory 'dir' => 'other'\n")
        self.assertContainsRe(d, b"=== modified file 'other/file'\n")
        self.assertContainsRe(d, b'--- old/dir/file\t')
        self.assertContainsRe(d, b'\\+\\+\\+ new/other/file\t')
        self.assertContainsRe(d, b'-contents\n\\+new contents\n')

    def test_renamed_directory(self):
        """Test when only a directory is only renamed."""
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/dir/'])
        self.build_tree_contents([('tree/dir/file', b'contents\n')])
        tree.add(['dir', 'dir/file'], ids=[b'dir-id', b'file-id'])
        tree.commit('one', rev_id=b'rev-1')
        tree.rename_one('dir', 'newdir')
        d = get_diff_as_string(tree.basis_tree(), tree)
        self.assertEqual(d, b"=== renamed directory 'dir' => 'newdir'\n")

    def test_renamed_file(self):
        """Test when a file is only renamed."""
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', b'contents\n')])
        tree.add(['file'], ids=[b'file-id'])
        tree.commit('one', rev_id=b'rev-1')
        tree.rename_one('file', 'newname')
        d = get_diff_as_string(tree.basis_tree(), tree)
        self.assertContainsRe(d, b"=== renamed file 'file' => 'newname'\n")
        self.assertNotContainsRe(d, b'---')

    def test_renamed_and_modified_file(self):
        """Test when a file is only renamed."""
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/file', b'contents\n')])
        tree.add(['file'], ids=[b'file-id'])
        tree.commit('one', rev_id=b'rev-1')
        tree.rename_one('file', 'newname')
        self.build_tree_contents([('tree/newname', b'new contents\n')])
        d = get_diff_as_string(tree.basis_tree(), tree)
        self.assertContainsRe(d, b"=== renamed file 'file' => 'newname'\n")
        self.assertContainsRe(d, b'--- old/file\t')
        self.assertContainsRe(d, b'\\+\\+\\+ new/newname\t')
        self.assertContainsRe(d, b'-contents\n\\+new contents\n')

    def test_internal_diff_exec_property(self):
        tree = self.make_branch_and_tree('tree')
        tt = tree.transform()
        tt.new_file('a', tt.root, [b'contents\n'], b'a-id', True)
        tt.new_file('b', tt.root, [b'contents\n'], b'b-id', False)
        tt.new_file('c', tt.root, [b'contents\n'], b'c-id', True)
        tt.new_file('d', tt.root, [b'contents\n'], b'd-id', False)
        tt.new_file('e', tt.root, [b'contents\n'], b'control-e-id', True)
        tt.new_file('f', tt.root, [b'contents\n'], b'control-f-id', False)
        tt.apply()
        tree.commit('one', rev_id=b'rev-1')
        tt = tree.transform()
        tt.set_executability(False, tt.trans_id_file_id(b'a-id'))
        tt.set_executability(True, tt.trans_id_file_id(b'b-id'))
        tt.set_executability(False, tt.trans_id_file_id(b'c-id'))
        tt.set_executability(True, tt.trans_id_file_id(b'd-id'))
        tt.apply()
        tree.rename_one('c', 'new-c')
        tree.rename_one('d', 'new-d')
        d = get_diff_as_string(tree.basis_tree(), tree)
        self.assertContainsRe(d, b"file 'a'.*\\(properties changed:.*\\+x to -x.*\\)")
        self.assertContainsRe(d, b"file 'b'.*\\(properties changed:.*-x to \\+x.*\\)")
        self.assertContainsRe(d, b"file 'c'.*\\(properties changed:.*\\+x to -x.*\\)")
        self.assertContainsRe(d, b"file 'd'.*\\(properties changed:.*-x to \\+x.*\\)")
        self.assertNotContainsRe(d, b"file 'e'")
        self.assertNotContainsRe(d, b"file 'f'")

    def test_binary_unicode_filenames(self):
        """Test that contents of files are *not* encoded in UTF-8 when there
        is a binary file in the diff.
        """
        self.requireFeature(features.UnicodeFilenameFeature)
        tree = self.make_branch_and_tree('tree')
        alpha, omega = ('α', 'ω')
        alpha_utf8, omega_utf8 = (alpha.encode('utf8'), omega.encode('utf8'))
        self.build_tree_contents([('tree/' + alpha, b'\x00'), ('tree/' + omega, b'The %s and the %s\n' % (alpha_utf8, omega_utf8))])
        tree.add([alpha])
        tree.add([omega])
        diff_content = StubO()
        diff.show_diff_trees(tree.basis_tree(), tree, diff_content)
        diff_content.check_types(self, bytes)
        d = b''.join(diff_content.write_record)
        self.assertContainsRe(d, b"=== added file '%s'" % alpha_utf8)
        self.assertContainsRe(d, b'Binary files a/%s.*and b/%s.* differ\n' % (alpha_utf8, alpha_utf8))
        self.assertContainsRe(d, b"=== added file '%s'" % omega_utf8)
        self.assertContainsRe(d, b'--- a/%s' % (omega_utf8,))
        self.assertContainsRe(d, b'\\+\\+\\+ b/%s' % (omega_utf8,))

    def test_unicode_filename(self):
        """Test when the filename are unicode."""
        self.requireFeature(features.UnicodeFilenameFeature)
        alpha, omega = ('α', 'ω')
        autf8, outf8 = (alpha.encode('utf8'), omega.encode('utf8'))
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/ren_' + alpha, b'contents\n')])
        tree.add(['ren_' + alpha], ids=[b'file-id-2'])
        self.build_tree_contents([('tree/del_' + alpha, b'contents\n')])
        tree.add(['del_' + alpha], ids=[b'file-id-3'])
        self.build_tree_contents([('tree/mod_' + alpha, b'contents\n')])
        tree.add(['mod_' + alpha], ids=[b'file-id-4'])
        tree.commit('one', rev_id=b'rev-1')
        tree.rename_one('ren_' + alpha, 'ren_' + omega)
        tree.remove('del_' + alpha)
        self.build_tree_contents([('tree/add_' + alpha, b'contents\n')])
        tree.add(['add_' + alpha], ids=[b'file-id'])
        self.build_tree_contents([('tree/mod_' + alpha, b'contents_mod\n')])
        d = get_diff_as_string(tree.basis_tree(), tree)
        self.assertContainsRe(d, b"=== renamed file 'ren_%s' => 'ren_%s'\n" % (autf8, outf8))
        self.assertContainsRe(d, b"=== added file 'add_%s'" % autf8)
        self.assertContainsRe(d, b"=== modified file 'mod_%s'" % autf8)
        self.assertContainsRe(d, b"=== removed file 'del_%s'" % autf8)

    def test_unicode_filename_path_encoding(self):
        """Test for bug #382699: unicode filenames on Windows should be shown
        in user encoding.
        """
        self.requireFeature(features.UnicodeFilenameFeature)
        _russian_test = 'Тест'
        directory = _russian_test + '/'
        test_txt = _russian_test + '.txt'
        u1234 = 'ሴ.txt'
        tree = self.make_branch_and_tree('.')
        self.build_tree_contents([(test_txt, b'foo\n'), (u1234, b'foo\n'), (directory, None)])
        tree.add([test_txt, u1234, directory])
        sio = BytesIO()
        diff.show_diff_trees(tree.basis_tree(), tree, sio, path_encoding='cp1251')
        output = subst_dates(sio.getvalue())
        shouldbe = b"=== added directory '%(directory)s'\n=== added file '%(test_txt)s'\n--- a/%(test_txt)s\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ b/%(test_txt)s\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -0,0 +1,1 @@\n+foo\n\n=== added file '?.txt'\n--- a/?.txt\tYYYY-MM-DD HH:MM:SS +ZZZZ\n+++ b/?.txt\tYYYY-MM-DD HH:MM:SS +ZZZZ\n@@ -0,0 +1,1 @@\n+foo\n\n" % {b'directory': _russian_test.encode('cp1251'), b'test_txt': test_txt.encode('cp1251')}
        self.assertEqualDiff(output, shouldbe)