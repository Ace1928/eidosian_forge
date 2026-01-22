import gzip
import os
import tarfile
import time
import zipfile
from io import BytesIO
from .. import errors, export, tests
from ..archive.tar import tarball_generator
from ..export import get_root_name
from . import features
class TestDirExport(tests.TestCaseWithTransport):

    def test_missing_file(self):
        self.build_tree(['a/', 'a/b', 'a/c'])
        wt = self.make_branch_and_tree('.')
        wt.add(['a', 'a/b', 'a/c'])
        os.unlink('a/c')
        export.export(wt, 'target', format='dir')
        self.assertPathExists('target/a/b')
        self.assertPathDoesNotExist('target/a/c')

    def test_empty(self):
        wt = self.make_branch_and_tree('.')
        export.export(wt, 'target', format='dir')
        self.assertEqual([], os.listdir('target'))

    def test_symlink(self):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        wt = self.make_branch_and_tree('.')
        os.symlink('source', 'link')
        wt.add(['link'])
        export.export(wt, 'target', format='dir')
        self.assertPathExists('target/link')

    def test_nested_tree(self):
        wt = self.make_branch_and_tree('.', format='development-subtree')
        subtree = self.make_branch_and_tree('subtree')
        self.build_tree(['subtree/file'])
        subtree.add(['file'])
        wt.add(['subtree'])
        export.export(wt, 'target', format='dir')
        self.assertPathExists('target/subtree')

    def test_to_existing_empty_dir_success(self):
        self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c'])
        wt = self.make_branch_and_tree('source')
        wt.add(['a', 'b', 'b/c'])
        wt.commit('1')
        self.build_tree(['target/'])
        export.export(wt, 'target', format='dir')
        self.assertPathExists('target/a')
        self.assertPathExists('target/b')
        self.assertPathExists('target/b/c')

    def test_empty_subdir(self):
        self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c'])
        wt = self.make_branch_and_tree('source')
        wt.add(['a', 'b', 'b/c'])
        wt.commit('1')
        self.build_tree(['target/'])
        export.export(wt, 'target', format='dir', subdir='')
        self.assertPathExists('target/a')
        self.assertPathExists('target/b')
        self.assertPathExists('target/b/c')

    def test_to_existing_nonempty_dir_fail(self):
        self.build_tree(['source/', 'source/a', 'source/b/', 'source/b/c'])
        wt = self.make_branch_and_tree('source')
        wt.add(['a', 'b', 'b/c'])
        wt.commit('1')
        self.build_tree(['target/', 'target/foo'])
        self.assertRaises(errors.BzrError, export.export, wt, 'target', format='dir')

    def test_existing_single_file(self):
        self.build_tree(['dir1/', 'dir1/dir2/', 'dir1/first', 'dir1/dir2/second'])
        wtree = self.make_branch_and_tree('dir1')
        wtree.add(['dir2', 'first', 'dir2/second'])
        wtree.commit('1')
        export.export(wtree, 'target1', format='dir', subdir='first')
        self.assertPathExists('target1/first')
        export.export(wtree, 'target2', format='dir', subdir='dir2/second')
        self.assertPathExists('target2/second')

    def test_files_same_timestamp(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('a', b'a-id', 'file', b'content\n'))])
        builder.build_snapshot(None, [('add', ('b', b'b-id', 'file', b'content\n'))])
        builder.finish_series()
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        tree = b.basis_tree()
        orig_iter_files_bytes = tree.iter_files_bytes

        def iter_files_bytes(to_fetch):
            for thing in orig_iter_files_bytes(to_fetch):
                yield thing
                time.sleep(1)
        tree.iter_files_bytes = iter_files_bytes
        export.export(tree, 'target', format='dir')
        t = self.get_transport('target')
        st_a = t.stat('a')
        st_b = t.stat('b')
        self.assertEqual(st_a.st_mtime, st_b.st_mtime)

    def test_files_per_file_timestamps(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        a_time = time.mktime((1999, 12, 12, 0, 0, 0, 0, 0, 0))
        b_time = time.mktime((1980, 1, 1, 0, 0, 0, 0, 0, 0))
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('a', b'a-id', 'file', b'content\n'))], timestamp=a_time)
        builder.build_snapshot(None, [('add', ('b', b'b-id', 'file', b'content\n'))], timestamp=b_time)
        builder.finish_series()
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        tree = b.basis_tree()
        export.export(tree, 'target', format='dir', per_file_timestamps=True)
        t = self.get_transport('target')
        self.assertEqual(a_time, t.stat('a').st_mtime)
        self.assertEqual(b_time, t.stat('b').st_mtime)

    def test_subdir_files_per_timestamps(self):
        builder = self.make_branch_builder('source')
        builder.start_series()
        foo_time = time.mktime((1999, 12, 12, 0, 0, 0, 0, 0, 0))
        builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('subdir', b'subdir-id', 'directory', '')), ('add', ('subdir/foo.txt', b'foo-id', 'file', b'content\n'))], timestamp=foo_time)
        builder.finish_series()
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        tree = b.basis_tree()
        export.export(tree, 'target', format='dir', subdir='subdir', per_file_timestamps=True)
        t = self.get_transport('target')
        self.assertEqual(foo_time, t.stat('foo.txt').st_mtime)