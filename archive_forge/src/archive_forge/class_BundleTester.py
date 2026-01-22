import bz2
import os
import sys
import tempfile
from io import BytesIO
from ... import diff, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...tests import features, test_commit
from ...tree import InterTree
from .. import bzrdir, inventory, knitrepo
from ..bundle.apply_bundle import install_bundle, merge_bundle
from ..bundle.bundle_data import BundleTree
from ..bundle.serializer import read_bundle, v4, v09, write_bundle
from ..bundle.serializer.v4 import BundleSerializerV4
from ..bundle.serializer.v08 import BundleSerializerV08
from ..bundle.serializer.v09 import BundleSerializerV09
from ..inventorytree import InventoryTree
class BundleTester:

    def bzrdir_format(self):
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = knitrepo.RepositoryFormatKnit1()
        return format

    def make_branch_and_tree(self, path, format=None):
        if format is None:
            format = self.bzrdir_format()
        return tests.TestCaseWithTransport.make_branch_and_tree(self, path, format)

    def make_branch(self, path, format=None, name=None):
        if format is None:
            format = self.bzrdir_format()
        return tests.TestCaseWithTransport.make_branch(self, path, format, name=name)

    def create_bundle_text(self, base_rev_id, rev_id):
        bundle_txt = BytesIO()
        rev_ids = write_bundle(self.b1.repository, rev_id, base_rev_id, bundle_txt, format=self.format)
        bundle_txt.seek(0)
        self.assertEqual(bundle_txt.readline(), b'# Bazaar revision bundle v%s\n' % self.format.encode('ascii'))
        self.assertEqual(bundle_txt.readline(), b'#\n')
        rev = self.b1.repository.get_revision(rev_id)
        self.assertEqual(bundle_txt.readline().decode('utf-8'), '# message:\n')
        bundle_txt.seek(0)
        return (bundle_txt, rev_ids)

    def get_valid_bundle(self, base_rev_id, rev_id, checkout_dir=None):
        """Create a bundle from base_rev_id -> rev_id in built-in branch.
        Make sure that the text generated is valid, and that it
        can be applied against the base, and generate the same information.

        :return: The in-memory bundle
        """
        bundle_txt, rev_ids = self.create_bundle_text(base_rev_id, rev_id)
        bundle = read_bundle(bundle_txt)
        repository = self.b1.repository
        for bundle_rev in bundle.real_revisions:
            branch_rev = repository.get_revision(bundle_rev.revision_id)
            for a in ('inventory_sha1', 'revision_id', 'parent_ids', 'timestamp', 'timezone', 'message', 'committer', 'parent_ids', 'properties'):
                self.assertEqual(getattr(branch_rev, a), getattr(bundle_rev, a))
            self.assertEqual(len(branch_rev.parent_ids), len(bundle_rev.parent_ids))
        self.assertEqual(rev_ids, [r.revision_id for r in bundle.real_revisions])
        self.valid_apply_bundle(base_rev_id, bundle, checkout_dir=checkout_dir)
        return bundle

    def get_invalid_bundle(self, base_rev_id, rev_id):
        """Create a bundle from base_rev_id -> rev_id in built-in branch.
        Munge the text so that it's invalid.

        :return: The in-memory bundle
        """
        bundle_txt, rev_ids = self.create_bundle_text(base_rev_id, rev_id)
        new_text = bundle_txt.getvalue().replace(b'executable:no', b'executable:yes')
        bundle_txt = BytesIO(new_text)
        bundle = read_bundle(bundle_txt)
        self.valid_apply_bundle(base_rev_id, bundle)
        return bundle

    def test_non_bundle(self):
        self.assertRaises(errors.NotABundle, read_bundle, BytesIO(b'#!/bin/sh\n'))

    def test_malformed(self):
        self.assertRaises(errors.BadBundle, read_bundle, BytesIO(b'# Bazaar revision bundle v'))

    def test_crlf_bundle(self):
        try:
            read_bundle(BytesIO(b'# Bazaar revision bundle v0.8\r\n'))
        except errors.BadBundle:
            pass

    def get_checkout(self, rev_id, checkout_dir=None):
        """Get a new tree, with the specified revision in it.
        """
        if checkout_dir is None:
            checkout_dir = tempfile.mkdtemp(prefix='test-branch-', dir='.')
        elif not os.path.exists(checkout_dir):
            os.mkdir(checkout_dir)
        tree = self.make_branch_and_tree(checkout_dir)
        s = BytesIO()
        ancestors = write_bundle(self.b1.repository, rev_id, b'null:', s, format=self.format)
        s.seek(0)
        self.assertIsInstance(s.getvalue(), bytes)
        install_bundle(tree.branch.repository, read_bundle(s))
        for ancestor in ancestors:
            old = self.b1.repository.revision_tree(ancestor)
            new = tree.branch.repository.revision_tree(ancestor)
            with old.lock_read(), new.lock_read():
                delta = new.changes_from(old)
                self.assertFalse(delta.has_changed(), 'Revision %s not copied correctly.' % (ancestor,))
                for path in old.all_versioned_paths():
                    try:
                        old_file = old.get_file(path)
                    except _mod_transport.NoSuchFile:
                        continue
                    self.assertEqual(old_file.read(), new.get_file(path).read())
        if not _mod_revision.is_null(rev_id):
            tree.branch.generate_revision_history(rev_id)
            tree.update()
            delta = tree.changes_from(self.b1.repository.revision_tree(rev_id))
            self.assertFalse(delta.has_changed(), 'Working tree has modifications: %s' % delta)
        return tree

    def valid_apply_bundle(self, base_rev_id, info, checkout_dir=None):
        """Get the base revision, apply the changes, and make
        sure everything matches the builtin branch.
        """
        to_tree = self.get_checkout(base_rev_id, checkout_dir=checkout_dir)
        to_tree.lock_write()
        try:
            self._valid_apply_bundle(base_rev_id, info, to_tree)
        finally:
            to_tree.unlock()

    def _valid_apply_bundle(self, base_rev_id, info, to_tree):
        original_parents = to_tree.get_parent_ids()
        repository = to_tree.branch.repository
        original_parents = to_tree.get_parent_ids()
        self.assertIs(repository.has_revision(base_rev_id), True)
        for rev in info.real_revisions:
            self.assertTrue(not repository.has_revision(rev.revision_id), 'Revision {%s} present before applying bundle' % rev.revision_id)
        merge_bundle(info, to_tree, True, merge.Merge3Merger, False, False)
        for rev in info.real_revisions:
            self.assertTrue(repository.has_revision(rev.revision_id), 'Missing revision {%s} after applying bundle' % rev.revision_id)
        self.assertTrue(to_tree.branch.repository.has_revision(info.target))
        self.assertEqual(original_parents + [info.target], to_tree.get_parent_ids())
        rev = info.real_revisions[-1]
        base_tree = self.b1.repository.revision_tree(rev.revision_id)
        to_tree = to_tree.branch.repository.revision_tree(rev.revision_id)
        base_files = list(base_tree.list_files())
        to_files = list(to_tree.list_files())
        self.assertEqual(len(base_files), len(to_files))
        for base_file, to_file in zip(base_files, to_files):
            self.assertEqual(base_file, to_file)
        for path, status, kind, entry in base_files:
            to_path = InterTree.get(base_tree, to_tree).find_target_path(path)
            self.assertEqual(base_tree.get_file_size(path), to_tree.get_file_size(to_path))
            self.assertEqual(base_tree.get_file_sha1(path), to_tree.get_file_sha1(to_path))

    def test_bundle(self):
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        self.build_tree_contents([('b1/one', b'one\n')])
        self.tree1.add('one', ids=b'one-id')
        self.tree1.set_root_id(b'root-id')
        self.tree1.commit('add one', rev_id=b'a@cset-0-1')
        bundle = self.get_valid_bundle(b'null:', b'a@cset-0-1')
        self.build_tree(['b1/with space.txt', 'b1/dir/', 'b1/dir/filein subdir.c', 'b1/dir/WithCaps.txt', 'b1/dir/ pre space', 'b1/sub/', 'b1/sub/sub/', 'b1/sub/sub/nonempty.txt'])
        self.build_tree_contents([('b1/sub/sub/emptyfile.txt', b''), ('b1/dir/nolastnewline.txt', b'bloop')])
        tt = self.tree1.transform()
        tt.new_file('executable', tt.root, [b'#!/bin/sh\n'], b'exe-1', True)
        tt.apply()
        self.tree1.add('with space.txt', ids=b'withspace-id')
        self.tree1.add(['dir', 'dir/filein subdir.c', 'dir/WithCaps.txt', 'dir/ pre space', 'dir/nolastnewline.txt', 'sub', 'sub/sub', 'sub/sub/nonempty.txt', 'sub/sub/emptyfile.txt'])
        self.tree1.commit('add whitespace', rev_id=b'a@cset-0-2')
        bundle = self.get_valid_bundle(b'a@cset-0-1', b'a@cset-0-2')
        bundle = self.get_valid_bundle(b'null:', b'a@cset-0-2')
        self.tree1.remove(['sub/sub/nonempty.txt', 'sub/sub/emptyfile.txt', 'sub/sub'])
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path('executable')
        tt.set_executability(False, trans_id)
        tt.apply()
        self.tree1.commit('removed', rev_id=b'a@cset-0-3')
        bundle = self.get_valid_bundle(b'a@cset-0-2', b'a@cset-0-3')
        self.assertRaises((errors.TestamentMismatch, errors.VersionedFileInvalidChecksum, errors.BadBundle), self.get_invalid_bundle, b'a@cset-0-2', b'a@cset-0-3')
        bundle = self.get_valid_bundle(b'null:', b'a@cset-0-3')
        self.tree1.rename_one('dir', 'sub/dir')
        self.tree1.commit('rename dir', rev_id=b'a@cset-0-4')
        bundle = self.get_valid_bundle(b'a@cset-0-3', b'a@cset-0-4')
        bundle = self.get_valid_bundle(b'null:', b'a@cset-0-4')
        with open('b1/sub/dir/WithCaps.txt', 'ab') as f:
            f.write(b'\nAdding some text\n')
        with open('b1/sub/dir/ pre space', 'ab') as f:
            f.write(b'\r\nAdding some\r\nDOS format lines\r\n')
        with open('b1/sub/dir/nolastnewline.txt', 'ab') as f:
            f.write(b'\n')
        self.tree1.rename_one('sub/dir/ pre space', 'sub/ start space')
        self.tree1.commit('Modified files', rev_id=b'a@cset-0-5')
        bundle = self.get_valid_bundle(b'a@cset-0-4', b'a@cset-0-5')
        self.tree1.rename_one('sub/dir/WithCaps.txt', 'temp')
        self.tree1.rename_one('with space.txt', 'WithCaps.txt')
        self.tree1.rename_one('temp', 'with space.txt')
        self.tree1.commit('swap filenames', rev_id=b'a@cset-0-6', verbose=False)
        bundle = self.get_valid_bundle(b'a@cset-0-5', b'a@cset-0-6')
        other = self.get_checkout(b'a@cset-0-5')
        tree1_inv = get_inventory_text(self.tree1.branch.repository, b'a@cset-0-5')
        tree2_inv = get_inventory_text(other.branch.repository, b'a@cset-0-5')
        self.assertEqualDiff(tree1_inv, tree2_inv)
        other.rename_one('sub/dir/nolastnewline.txt', 'sub/nolastnewline.txt')
        other.commit('rename file', rev_id=b'a@cset-0-6b')
        self.tree1.merge_from_branch(other.branch)
        self.tree1.commit('Merge', rev_id=b'a@cset-0-7', verbose=False)
        bundle = self.get_valid_bundle(b'a@cset-0-6', b'a@cset-0-7')

    def _test_symlink_bundle(self, link_name, link_target, new_link_target):
        link_id = b'link-1'
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        tt = self.tree1.transform()
        tt.new_symlink(link_name, tt.root, link_target, link_id)
        tt.apply()
        self.tree1.commit('add symlink', rev_id=b'l@cset-0-1')
        bundle = self.get_valid_bundle(b'null:', b'l@cset-0-1')
        if getattr(bundle, 'revision_tree', None) is not None:
            bund_tree = bundle.revision_tree(self.b1.repository, b'l@cset-0-1')
            self.assertEqual(link_target, bund_tree.get_symlink_target(link_name))
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path(link_name)
        tt.adjust_path('link2', tt.root, trans_id)
        tt.delete_contents(trans_id)
        tt.create_symlink(new_link_target, trans_id)
        tt.apply()
        self.tree1.commit('rename and change symlink', rev_id=b'l@cset-0-2')
        bundle = self.get_valid_bundle(b'l@cset-0-1', b'l@cset-0-2')
        if getattr(bundle, 'revision_tree', None) is not None:
            bund_tree = bundle.revision_tree(self.b1.repository, b'l@cset-0-2')
            self.assertEqual(new_link_target, bund_tree.get_symlink_target('link2'))
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path('link2')
        tt.delete_contents(trans_id)
        tt.create_symlink('jupiter', trans_id)
        tt.apply()
        self.tree1.commit('just change symlink target', rev_id=b'l@cset-0-3')
        bundle = self.get_valid_bundle(b'l@cset-0-2', b'l@cset-0-3')
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path('link2')
        tt.delete_contents(trans_id)
        tt.apply()
        self.tree1.commit('Delete symlink', rev_id=b'l@cset-0-4')
        bundle = self.get_valid_bundle(b'l@cset-0-3', b'l@cset-0-4')

    def test_symlink_bundle(self):
        self._test_symlink_bundle('link', 'bar/foo', 'mars')

    def test_unicode_symlink_bundle(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_symlink_bundle('€link', 'bar/€foo', 'mars€')

    def test_binary_bundle(self):
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        tt = self.tree1.transform()
        tt.new_file('file', tt.root, [b'\x00\n\x00\r\x01\n\x02\r\xff'], b'binary-1')
        tt.new_file('file2', tt.root, [b'\x01\n\x02\r\x03\n\x04\r\xff'], b'binary-2')
        tt.apply()
        self.tree1.commit('add binary', rev_id=b'b@cset-0-1')
        self.get_valid_bundle(b'null:', b'b@cset-0-1')
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path('file')
        tt.delete_contents(trans_id)
        tt.apply()
        self.tree1.commit('delete binary', rev_id=b'b@cset-0-2')
        self.get_valid_bundle(b'b@cset-0-1', b'b@cset-0-2')
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path('file2')
        tt.adjust_path('file3', tt.root, trans_id)
        tt.delete_contents(trans_id)
        tt.create_file([b'file\rcontents\x00\n\x00'], trans_id)
        tt.apply()
        self.tree1.commit('rename and modify binary', rev_id=b'b@cset-0-3')
        self.get_valid_bundle(b'b@cset-0-2', b'b@cset-0-3')
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path('file3')
        tt.delete_contents(trans_id)
        tt.create_file([b'\x00file\rcontents'], trans_id)
        tt.apply()
        self.tree1.commit('just modify binary', rev_id=b'b@cset-0-4')
        self.get_valid_bundle(b'b@cset-0-3', b'b@cset-0-4')
        self.get_valid_bundle(b'null:', b'b@cset-0-4')

    def test_last_modified(self):
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        tt = self.tree1.transform()
        tt.new_file('file', tt.root, [b'file'], b'file')
        tt.apply()
        self.tree1.commit('create file', rev_id=b'a@lmod-0-1')
        tt = self.tree1.transform()
        trans_id = tt.trans_id_tree_path('file')
        tt.delete_contents(trans_id)
        tt.create_file([b'file2'], trans_id)
        tt.apply()
        self.tree1.commit('modify text', rev_id=b'a@lmod-0-2a')
        other = self.get_checkout(b'a@lmod-0-1')
        tt = other.transform()
        trans_id = tt.trans_id_tree_path('file2')
        tt.delete_contents(trans_id)
        tt.create_file([b'file2'], trans_id)
        tt.apply()
        other.commit('modify text in another tree', rev_id=b'a@lmod-0-2b')
        self.tree1.merge_from_branch(other.branch)
        self.tree1.commit('Merge', rev_id=b'a@lmod-0-3', verbose=False)
        self.tree1.commit('Merge', rev_id=b'a@lmod-0-4')
        bundle = self.get_valid_bundle(b'a@lmod-0-2a', b'a@lmod-0-4')

    def test_hide_history(self):
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        with open('b1/one', 'wb') as f:
            f.write(b'one\n')
        self.tree1.add('one')
        self.tree1.commit('add file', rev_id=b'a@cset-0-1')
        with open('b1/one', 'wb') as f:
            f.write(b'two\n')
        self.tree1.commit('modify', rev_id=b'a@cset-0-2')
        with open('b1/one', 'wb') as f:
            f.write(b'three\n')
        self.tree1.commit('modify', rev_id=b'a@cset-0-3')
        bundle_file = BytesIO()
        rev_ids = write_bundle(self.tree1.branch.repository, b'a@cset-0-3', b'a@cset-0-1', bundle_file, format=self.format)
        self.assertNotContainsRe(bundle_file.getvalue(), b'\x08two\x08')
        self.assertContainsRe(self.get_raw(bundle_file), b'one')
        self.assertContainsRe(self.get_raw(bundle_file), b'three')

    def test_bundle_same_basis(self):
        """Ensure using the basis as the target doesn't cause an error"""
        self.tree1 = self.make_branch_and_tree('b1')
        self.tree1.commit('add file', rev_id=b'a@cset-0-1')
        bundle_file = BytesIO()
        rev_ids = write_bundle(self.tree1.branch.repository, b'a@cset-0-1', b'a@cset-0-1', bundle_file)

    @staticmethod
    def get_raw(bundle_file):
        return bundle_file.getvalue()

    def test_unicode_bundle(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        os.mkdir('b1')
        f = open('b1/with Dod€', 'wb')
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        f.write('A file\nWith international man of mystery\nWilliam Dodé\n'.encode())
        f.close()
        self.tree1.add(['with Dod€'], ids=[b'withdod-id'])
        self.tree1.commit('i18n commit from William Dodé', rev_id=b'i18n-1', committer='William Dodé')
        bundle = self.get_valid_bundle(b'null:', b'i18n-1')
        f = open('b1/with Dod€', 'wb')
        f.write('Modified µ\n'.encode())
        f.close()
        self.tree1.commit('modified', rev_id=b'i18n-2')
        bundle = self.get_valid_bundle(b'i18n-1', b'i18n-2')
        self.tree1.rename_one('with Dod€', 'B€gfors')
        self.tree1.commit('renamed, the new i18n man', rev_id=b'i18n-3', committer='Erik Bågfors')
        bundle = self.get_valid_bundle(b'i18n-2', b'i18n-3')
        self.tree1.remove(['B€gfors'])
        self.tree1.commit('removed', rev_id=b'i18n-4')
        bundle = self.get_valid_bundle(b'i18n-3', b'i18n-4')
        bundle = self.get_valid_bundle(b'null:', b'i18n-4')

    def test_whitespace_bundle(self):
        if sys.platform in ('win32', 'cygwin'):
            raise tests.TestSkipped("Windows doesn't support filenames with tabs or trailing spaces")
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        self.build_tree(['b1/trailing space '])
        self.tree1.add(['trailing space '])
        self.tree1.commit('funky whitespace', rev_id=b'white-1')
        bundle = self.get_valid_bundle(b'null:', b'white-1')
        with open('b1/trailing space ', 'ab') as f:
            f.write(b'add some text\n')
        self.tree1.commit('add text', rev_id=b'white-2')
        bundle = self.get_valid_bundle(b'white-1', b'white-2')
        self.tree1.rename_one('trailing space ', ' start and end space ')
        self.tree1.commit('rename', rev_id=b'white-3')
        bundle = self.get_valid_bundle(b'white-2', b'white-3')
        self.tree1.remove([' start and end space '])
        self.tree1.commit('removed', rev_id=b'white-4')
        bundle = self.get_valid_bundle(b'white-3', b'white-4')
        bundle = self.get_valid_bundle(b'null:', b'white-4')

    def test_alt_timezone_bundle(self):
        self.tree1 = self.make_branch_and_memory_tree('b1')
        self.b1 = self.tree1.branch
        builder = treebuilder.TreeBuilder()
        self.tree1.lock_write()
        builder.start_tree(self.tree1)
        builder.build(['newfile'])
        builder.finish_tree()
        self.tree1.commit('non-hour offset timezone', rev_id=b'tz-1', timezone=19800, timestamp=1152544886.0)
        bundle = self.get_valid_bundle(b'null:', b'tz-1')
        rev = bundle.revisions[0]
        self.assertEqual('Mon 2006-07-10 20:51:26.000000000 +0530', rev.date)
        self.assertEqual(19800, rev.timezone)
        self.assertEqual(1152544886.0, rev.timestamp)
        self.tree1.unlock()

    def test_bundle_root_id(self):
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        self.tree1.commit('message', rev_id=b'revid1')
        bundle = self.get_valid_bundle(b'null:', b'revid1')
        tree = self.get_bundle_tree(bundle, b'revid1')
        root_revision = tree.get_file_revision('')
        self.assertEqual(b'revid1', root_revision)

    def test_install_revisions(self):
        self.tree1 = self.make_branch_and_tree('b1')
        self.b1 = self.tree1.branch
        self.tree1.commit('message', rev_id=b'rev2a')
        bundle = self.get_valid_bundle(b'null:', b'rev2a')
        branch2 = self.make_branch('b2')
        self.assertFalse(branch2.repository.has_revision(b'rev2a'))
        target_revision = bundle.install_revisions(branch2.repository)
        self.assertTrue(branch2.repository.has_revision(b'rev2a'))
        self.assertEqual(b'rev2a', target_revision)

    def test_bundle_empty_property(self):
        """Test serializing revision properties with an empty value."""
        tree = self.make_branch_and_memory_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add([''], ids=[b'TREE_ROOT'])
        tree.commit('One', revprops={'one': 'two', 'empty': ''}, rev_id=b'rev1')
        self.b1 = tree.branch
        bundle_sio, revision_ids = self.create_bundle_text(b'null:', b'rev1')
        bundle = read_bundle(bundle_sio)
        revision_info = bundle.revisions[0]
        self.assertEqual(b'rev1', revision_info.revision_id)
        rev = revision_info.as_revision()
        self.assertEqual({'branch-nick': 'tree', 'empty': '', 'one': 'two'}, rev.properties)

    def test_bundle_sorted_properties(self):
        """For stability the writer should write properties in sorted order."""
        tree = self.make_branch_and_memory_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add([''], ids=[b'TREE_ROOT'])
        tree.commit('One', rev_id=b'rev1', revprops={'a': '4', 'b': '3', 'c': '2', 'd': '1'})
        self.b1 = tree.branch
        bundle_sio, revision_ids = self.create_bundle_text(b'null:', b'rev1')
        bundle = read_bundle(bundle_sio)
        revision_info = bundle.revisions[0]
        self.assertEqual(b'rev1', revision_info.revision_id)
        rev = revision_info.as_revision()
        self.assertEqual({'branch-nick': 'tree', 'a': '4', 'b': '3', 'c': '2', 'd': '1'}, rev.properties)

    def test_bundle_unicode_properties(self):
        """We should be able to round trip a non-ascii property."""
        tree = self.make_branch_and_memory_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.add([''], ids=[b'TREE_ROOT'])
        tree.commit('One', rev_id=b'rev1', revprops={'omega': 'Ω', 'alpha': 'α'})
        self.b1 = tree.branch
        bundle_sio, revision_ids = self.create_bundle_text(b'null:', b'rev1')
        bundle = read_bundle(bundle_sio)
        revision_info = bundle.revisions[0]
        self.assertEqual(b'rev1', revision_info.revision_id)
        rev = revision_info.as_revision()
        self.assertEqual({'branch-nick': 'tree', 'omega': 'Ω', 'alpha': 'α'}, rev.properties)

    def test_bundle_with_ghosts(self):
        tree = self.make_branch_and_tree('tree')
        self.b1 = tree.branch
        self.build_tree_contents([('tree/file', b'content1')])
        tree.add(['file'])
        tree.commit('rev1')
        self.build_tree_contents([('tree/file', b'content2')])
        tree.add_parent_tree_id(b'ghost')
        tree.commit('rev2', rev_id=b'rev2')
        bundle = self.get_valid_bundle(b'null:', b'rev2')

    def make_simple_tree(self, format=None):
        tree = self.make_branch_and_tree('b1', format=format)
        self.b1 = tree.branch
        self.build_tree(['b1/file'])
        tree.add('file')
        return tree

    def test_across_serializers(self):
        tree = self.make_simple_tree('knit')
        tree.commit('hello', rev_id=b'rev1')
        tree.commit('hello', rev_id=b'rev2')
        bundle = read_bundle(self.create_bundle_text(b'null:', b'rev2')[0])
        repo = self.make_repository('repo', format='dirstate-with-subtree')
        bundle.install_revisions(repo)
        inv_text = b''.join(repo._get_inventory_xml(b'rev2'))
        self.assertNotContainsRe(inv_text, b'format="5"')
        self.assertContainsRe(inv_text, b'format="7"')

    def make_repo_with_installed_revisions(self):
        tree = self.make_simple_tree('knit')
        tree.commit('hello', rev_id=b'rev1')
        tree.commit('hello', rev_id=b'rev2')
        bundle = read_bundle(self.create_bundle_text(b'null:', b'rev2')[0])
        repo = self.make_repository('repo', format='dirstate-with-subtree')
        bundle.install_revisions(repo)
        return repo

    def test_across_models(self):
        repo = self.make_repo_with_installed_revisions()
        inv = repo.get_inventory(b'rev2')
        self.assertEqual(b'rev2', inv.root.revision)
        root_id = inv.root.file_id
        repo.lock_read()
        self.addCleanup(repo.unlock)
        self.assertEqual({(root_id, b'rev1'): (), (root_id, b'rev2'): ((root_id, b'rev1'),)}, repo.texts.get_parent_map([(root_id, b'rev1'), (root_id, b'rev2')]))

    def test_inv_hash_across_serializers(self):
        repo = self.make_repo_with_installed_revisions()
        recorded_inv_sha1 = repo.get_revision(b'rev2').inventory_sha1
        xml = b''.join(repo._get_inventory_xml(b'rev2'))
        self.assertEqual(osutils.sha_string(xml), recorded_inv_sha1)

    def test_across_models_incompatible(self):
        tree = self.make_simple_tree('dirstate-with-subtree')
        tree.commit('hello', rev_id=b'rev1')
        tree.commit('hello', rev_id=b'rev2')
        try:
            bundle = read_bundle(self.create_bundle_text(b'null:', b'rev1')[0])
        except errors.IncompatibleBundleFormat:
            raise tests.TestSkipped("Format 0.8 doesn't work with knit3")
        repo = self.make_repository('repo', format='knit')
        bundle.install_revisions(repo)
        bundle = read_bundle(self.create_bundle_text(b'null:', b'rev2')[0])
        self.assertRaises(errors.IncompatibleRevision, bundle.install_revisions, repo)

    def test_get_merge_request(self):
        tree = self.make_simple_tree()
        tree.commit('hello', rev_id=b'rev1')
        tree.commit('hello', rev_id=b'rev2')
        bundle = read_bundle(self.create_bundle_text(b'null:', b'rev1')[0])
        result = bundle.get_merge_request(tree.branch.repository)
        self.assertEqual((None, b'rev1', 'inapplicable'), result)

    def test_with_subtree(self):
        tree = self.make_branch_and_tree('tree', format='dirstate-with-subtree')
        self.b1 = tree.branch
        subtree = self.make_branch_and_tree('tree/subtree', format='dirstate-with-subtree')
        tree.add('subtree')
        tree.commit('hello', rev_id=b'rev1')
        try:
            bundle = read_bundle(self.create_bundle_text(b'null:', b'rev1')[0])
        except errors.IncompatibleBundleFormat:
            raise tests.TestSkipped("Format 0.8 doesn't work with knit3")
        if isinstance(bundle, v09.BundleInfo09):
            raise tests.TestSkipped("Format 0.9 doesn't work with subtrees")
        repo = self.make_repository('repo', format='knit')
        self.assertRaises(errors.IncompatibleRevision, bundle.install_revisions, repo)
        repo2 = self.make_repository('repo2', format='dirstate-with-subtree')
        bundle.install_revisions(repo2)

    def test_revision_id_with_slash(self):
        self.tree1 = self.make_branch_and_tree('tree')
        self.b1 = self.tree1.branch
        try:
            self.tree1.commit('Revision/id/with/slashes', rev_id=b'rev/id')
        except ValueError:
            raise tests.TestSkipped("Repository doesn't support revision ids with slashes")
        bundle = self.get_valid_bundle(b'null:', b'rev/id')

    def test_skip_file(self):
        """Make sure we don't accidentally write to the wrong versionedfile"""
        self.tree1 = self.make_branch_and_tree('tree')
        self.b1 = self.tree1.branch
        self.build_tree_contents([('tree/file2', b'contents1')])
        self.tree1.add('file2', ids=b'file2-id')
        self.tree1.commit('rev1', rev_id=b'reva')
        self.build_tree_contents([('tree/file3', b'contents2')])
        self.tree1.add('file3', ids=b'file3-id')
        rev2 = self.tree1.commit('rev2')
        target = self.tree1.controldir.sprout('target').open_workingtree()
        self.build_tree_contents([('tree/file2', b'contents3')])
        self.tree1.commit('rev3', rev_id=b'rev3')
        bundle = self.get_valid_bundle(b'reva', b'rev3')
        if getattr(bundle, 'get_bundle_reader', None) is None:
            raise tests.TestSkipped('Bundle format cannot provide reader')
        file_ids = {(f, r) for b, m, k, r, f in bundle.get_bundle_reader().iter_records() if f is not None}
        self.assertEqual({(b'file2-id', b'rev3'), (b'file3-id', rev2)}, file_ids)
        bundle.install_revisions(target.branch.repository)