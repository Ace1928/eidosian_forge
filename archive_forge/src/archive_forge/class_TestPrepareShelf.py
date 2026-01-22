import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
class TestPrepareShelf(tests.TestCaseWithTransport):

    def prepare_shelve_rename(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        tree.add(['foo'], ids=[b'foo-id'])
        tree.commit('foo')
        tree.rename_one('foo', 'bar')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('rename', b'foo-id', 'foo', 'bar')], list(creator.iter_shelvable()))
        return creator

    def check_shelve_rename(self, creator):
        work_trans_id = creator.work_transform.trans_id_file_id(b'foo-id')
        self.assertEqual('foo', creator.work_transform.final_name(work_trans_id))
        shelf_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
        self.assertEqual('bar', creator.shelf_transform.final_name(shelf_trans_id))

    def test_shelve_rename(self):
        creator = self.prepare_shelve_rename()
        creator.shelve_rename(b'foo-id')
        self.check_shelve_rename(creator)

    def test_shelve_change_handles_rename(self):
        creator = self.prepare_shelve_rename()
        creator.shelve_change(('rename', b'foo-id', 'foo', 'bar'))
        self.check_shelve_rename(creator)

    def prepare_shelve_move(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo/', 'bar/', 'foo/baz'])
        tree.add(['foo', 'bar', 'foo/baz'], ids=[b'foo-id', b'bar-id', b'baz-id'])
        tree.commit('foo')
        tree.rename_one('foo/baz', 'bar/baz')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('rename', b'baz-id', 'foo/baz', 'bar/baz')], list(creator.iter_shelvable()))
        return (creator, tree)

    def check_shelve_move(self, creator, tree):
        work_trans_id = creator.work_transform.trans_id_file_id(b'baz-id')
        work_foo = creator.work_transform.trans_id_file_id(b'foo-id')
        self.assertEqual(work_foo, creator.work_transform.final_parent(work_trans_id))
        shelf_trans_id = creator.shelf_transform.trans_id_file_id(b'baz-id')
        shelf_bar = creator.shelf_transform.trans_id_file_id(b'bar-id')
        self.assertEqual(shelf_bar, creator.shelf_transform.final_parent(shelf_trans_id))
        creator.transform()
        self.assertEqual('foo/baz', tree.id2path(b'baz-id'))

    def test_shelve_move(self):
        creator, tree = self.prepare_shelve_move()
        creator.shelve_rename(b'baz-id')
        self.check_shelve_move(creator, tree)

    def test_shelve_change_handles_move(self):
        creator, tree = self.prepare_shelve_move()
        creator.shelve_change(('rename', b'baz-id', 'foo/baz', 'bar/baz'))
        self.check_shelve_move(creator, tree)

    def test_shelve_changed_root_id(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        tree.set_root_id(b'first-root-id')
        tree.add(['foo'], ids=[b'foo-id'])
        tree.commit('foo')
        tree.set_root_id(b'second-root-id')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.expectFailure("shelf doesn't support shelving root changes yet", self.assertEqual, [('delete file', b'first-root-id', 'directory', ''), ('add file', b'second-root-id', 'directory', ''), ('rename', b'foo-id', 'foo', 'foo')], list(creator.iter_shelvable()))
        self.assertEqual([('delete file', b'first-root-id', 'directory', ''), ('add file', b'second-root-id', 'directory', ''), ('rename', b'foo-id', 'foo', 'foo')], list(creator.iter_shelvable()))

    def assertShelvedFileEqual(self, expected_content, creator, file_id):
        s_trans_id = creator.shelf_transform.trans_id_file_id(file_id)
        shelf_file = creator.shelf_transform._limbo_name(s_trans_id)
        self.assertFileEqual(expected_content, shelf_file)

    def prepare_content_change(self):
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree_contents([('foo', b'a\n')])
        tree.add('foo', ids=b'foo-id')
        tree.commit('Committed foo')
        self.build_tree_contents([('foo', b'b\na\nc\n')])
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        return creator

    def test_shelve_content_change(self):
        creator = self.prepare_content_change()
        self.assertEqual([('modify text', b'foo-id')], list(creator.iter_shelvable()))
        creator.shelve_lines(b'foo-id', [b'a\n', b'c\n'])
        creator.transform()
        self.assertFileEqual(b'a\nc\n', 'foo')
        self.assertShelvedFileEqual(b'b\na\n', creator, b'foo-id')

    def test_shelve_change_handles_modify_text(self):
        creator = self.prepare_content_change()
        creator.shelve_change(('modify text', b'foo-id'))
        creator.transform()
        self.assertFileEqual(b'a\n', 'foo')
        self.assertShelvedFileEqual(b'b\na\nc\n', creator, b'foo-id')

    def test_shelve_all(self):
        creator = self.prepare_content_change()
        creator.shelve_all()
        creator.transform()
        self.assertFileEqual(b'a\n', 'foo')
        self.assertShelvedFileEqual(b'b\na\nc\n', creator, b'foo-id')

    def prepare_shelve_creation(self):
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.commit('Empty tree')
        self.build_tree_contents([('foo', b'a\n'), ('bar/',)])
        tree.add(['foo', 'bar'], ids=[b'foo-id', b'bar-id'])
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('add file', b'bar-id', 'directory', 'bar'), ('add file', b'foo-id', 'file', 'foo')], sorted(list(creator.iter_shelvable())))
        return (creator, tree)

    def check_shelve_creation(self, creator, tree):
        self.assertRaises(StopIteration, next, tree.iter_entries_by_dir(specific_files=['foo']))
        s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
        self.assertEqual(b'foo-id', creator.shelf_transform.final_file_id(s_trans_id))
        self.assertPathDoesNotExist('foo')
        self.assertPathDoesNotExist('bar')
        self.assertShelvedFileEqual('a\n', creator, b'foo-id')
        s_bar_trans_id = creator.shelf_transform.trans_id_file_id(b'bar-id')
        self.assertEqual('directory', creator.shelf_transform.final_kind(s_bar_trans_id))

    def test_shelve_creation(self):
        creator, tree = self.prepare_shelve_creation()
        creator.shelve_creation(b'foo-id')
        creator.shelve_creation(b'bar-id')
        creator.transform()
        self.check_shelve_creation(creator, tree)

    def test_shelve_change_handles_creation(self):
        creator, tree = self.prepare_shelve_creation()
        creator.shelve_change(('add file', b'foo-id', 'file', 'foo'))
        creator.shelve_change(('add file', b'bar-id', 'directory', 'bar'))
        creator.transform()
        self.check_shelve_creation(creator, tree)

    def test_shelve_directory_with_ignored(self):
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.commit('Empty tree')
        self.build_tree_contents([('foo', b'a\n'), ('bar/',), ('bar/ignored', b'ign\n')])
        tree.add(['foo', 'bar'], ids=[b'foo-id', b'bar-id'])
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('add file', b'bar-id', 'directory', 'bar'), ('add file', b'foo-id', 'file', 'foo')], sorted(list(creator.iter_shelvable())))
        ignores._set_user_ignores([])
        in_patterns = ['ignored']
        ignores.add_unique_user_ignores(in_patterns)
        creator.shelve_change(('add file', b'bar-id', 'directory', 'bar'))
        try:
            creator.transform()
            self.check_shelve_creation(creator, tree)
        except transform.MalformedTransform:
            raise KnownFailure('shelving directory with ignored file: see bug #611739')

    def _test_shelve_symlink_creation(self, link_name, link_target, shelve_change=False):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.commit('Empty tree')
        os.symlink(link_target, link_name)
        tree.add(link_name, ids=b'foo-id')
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('add file', b'foo-id', 'symlink', link_name)], list(creator.iter_shelvable()))
        if shelve_change:
            creator.shelve_change(('add file', b'foo-id', 'symlink', link_name))
        else:
            creator.shelve_creation(b'foo-id')
        creator.transform()
        s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
        self.assertPathDoesNotExist(link_name)
        limbo_name = creator.shelf_transform._limbo_name(s_trans_id)
        self.assertEqual(link_target, osutils.readlink(limbo_name))
        ptree = creator.shelf_transform.get_preview_tree()
        self.assertEqual(link_target, ptree.get_symlink_target(ptree.id2path(b'foo-id')))

    def test_shelve_symlink_creation(self):
        self._test_shelve_symlink_creation('foo', 'bar')

    def test_shelve_unicode_symlink_creation(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_shelve_symlink_creation('fo€o', 'b€ar')

    def test_shelve_change_handles_symlink_creation(self):
        self._test_shelve_symlink_creation('foo', 'bar', shelve_change=True)

    def _test_shelve_symlink_target_change(self, link_name, old_target, new_target, shelve_change=False):
        self.requireFeature(features.SymlinkFeature(self.test_dir))
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        os.symlink(old_target, link_name)
        tree.add(link_name, ids=b'foo-id')
        tree.commit('commit symlink')
        os.unlink(link_name)
        os.symlink(new_target, link_name)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('modify target', b'foo-id', link_name, old_target, new_target)], list(creator.iter_shelvable()))
        if shelve_change:
            creator.shelve_change(('modify target', b'foo-id', link_name, old_target, new_target))
        else:
            creator.shelve_modify_target(b'foo-id')
        creator.transform()
        self.assertEqual(old_target, osutils.readlink(link_name))
        s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
        limbo_name = creator.shelf_transform._limbo_name(s_trans_id)
        self.assertEqual(new_target, osutils.readlink(limbo_name))
        ptree = creator.shelf_transform.get_preview_tree()
        self.assertEqual(new_target, ptree.get_symlink_target(ptree.id2path(b'foo-id')))

    def test_shelve_symlink_target_change(self):
        self._test_shelve_symlink_target_change('foo', 'bar', 'baz')

    def test_shelve_unicode_symlink_target_change(self):
        self.requireFeature(features.UnicodeFilenameFeature)
        self._test_shelve_symlink_target_change('fo€o', 'b€ar', 'b€az')

    def test_shelve_change_handles_symlink_target_change(self):
        self._test_shelve_symlink_target_change('foo', 'bar', 'baz', shelve_change=True)

    def test_shelve_creation_no_contents(self):
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        tree.commit('Empty tree')
        self.build_tree(['foo'])
        tree.add('foo', ids=b'foo-id')
        os.unlink('foo')
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('add file', b'foo-id', None, 'foo')], sorted(list(creator.iter_shelvable())))
        creator.shelve_creation(b'foo-id')
        creator.transform()
        self.assertRaises(StopIteration, next, tree.iter_entries_by_dir(specific_files=['foo']))
        self.assertShelvedFileEqual('', creator, b'foo-id')
        s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
        self.assertEqual(b'foo-id', creator.shelf_transform.final_file_id(s_trans_id))
        self.assertPathDoesNotExist('foo')

    def prepare_shelve_deletion(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree_contents([('tree/foo/',), ('tree/foo/bar', b'baz')])
        tree.add(['foo', 'foo/bar'], ids=[b'foo-id', b'bar-id'])
        tree.commit('Added file and directory')
        tree.unversion(['foo', 'foo/bar'])
        os.unlink('tree/foo/bar')
        os.rmdir('tree/foo')
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('delete file', b'bar-id', 'file', 'foo/bar'), ('delete file', b'foo-id', 'directory', 'foo')], sorted(list(creator.iter_shelvable())))
        return (creator, tree)

    def check_shelve_deletion(self, tree):
        self.assertEqual(tree.id2path(b'foo-id'), 'foo')
        self.assertEqual(tree.id2path(b'bar-id'), 'foo/bar')
        self.assertFileEqual(b'baz', 'tree/foo/bar')

    def test_shelve_deletion(self):
        creator, tree = self.prepare_shelve_deletion()
        creator.shelve_deletion(b'foo-id')
        creator.shelve_deletion(b'bar-id')
        creator.transform()
        self.check_shelve_deletion(tree)

    def test_shelve_change_handles_deletion(self):
        creator, tree = self.prepare_shelve_deletion()
        creator.shelve_change(('delete file', b'foo-id', 'directory', 'foo'))
        creator.shelve_change(('delete file', b'bar-id', 'file', 'foo/bar'))
        creator.transform()
        self.check_shelve_deletion(tree)

    def test_shelve_delete_contents(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo'])
        tree.add('foo', ids=b'foo-id')
        tree.commit('Added file and directory')
        os.unlink('tree/foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('delete file', b'foo-id', 'file', 'foo')], sorted(list(creator.iter_shelvable())))
        creator.shelve_deletion(b'foo-id')
        creator.transform()
        self.assertPathExists('tree/foo')

    def prepare_shelve_change_kind(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/foo', b'bar')])
        tree.add('foo', ids=b'foo-id')
        tree.commit('Added file and directory')
        os.unlink('tree/foo')
        os.mkdir('tree/foo')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('change kind', b'foo-id', 'file', 'directory', 'foo')], sorted(list(creator.iter_shelvable())))
        return creator

    def check_shelve_change_kind(self, creator):
        self.assertFileEqual(b'bar', 'tree/foo')
        s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
        self.assertEqual('directory', creator.shelf_transform._new_contents[s_trans_id])

    def test_shelve_change_kind(self):
        creator = self.prepare_shelve_change_kind()
        creator.shelve_content_change(b'foo-id')
        creator.transform()
        self.check_shelve_change_kind(creator)

    def test_shelve_change_handles_change_kind(self):
        creator = self.prepare_shelve_change_kind()
        creator.shelve_change(('change kind', b'foo-id', 'file', 'directory', 'foo'))
        creator.transform()
        self.check_shelve_change_kind(creator)

    def test_shelve_change_unknown_change(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        e = self.assertRaises(ValueError, creator.shelve_change, ('unknown',))
        self.assertEqual('Unknown change kind: "unknown"', str(e))

    def test_shelve_unversion(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo'])
        tree.add('foo', ids=b'foo-id')
        tree.commit('Added file and directory')
        tree.unversion(['foo'])
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([('delete file', b'foo-id', 'file', 'foo')], sorted(list(creator.iter_shelvable())))
        creator.shelve_deletion(b'foo-id')
        creator.transform()
        self.assertPathExists('tree/foo')

    def test_shelve_serialization(self):
        tree = self.make_branch_and_tree('.')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        shelf_file = open('shelf', 'wb')
        self.addCleanup(shelf_file.close)
        try:
            creator.write_shelf(shelf_file)
        finally:
            shelf_file.close()
        self.assertFileEqual(EMPTY_SHELF, 'shelf')

    def test_write_shelf(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/foo'])
        tree.add('foo', ids=b'foo-id')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        list(creator.iter_shelvable())
        creator.shelve_creation(b'foo-id')
        with open('shelf', 'wb') as shelf_file:
            creator.write_shelf(shelf_file)
        parser = pack.ContainerPushParser()
        with open('shelf', 'rb') as shelf_file:
            parser.accept_bytes(shelf_file.read())
        tt = tree.preview_transform()
        self.addCleanup(tt.finalize)
        records = iter(parser.read_pending_records())
        next(records)
        tt.deserialize(records)

    def test_shelve_unversioned(self):
        tree = self.make_branch_and_tree('tree')
        with tree.lock_tree_write():
            self.assertRaises(errors.PathsNotVersionedError, shelf.ShelfCreator, tree, tree.basis_tree(), ['foo'])
        wt = workingtree.WorkingTree.open('tree')
        wt.lock_tree_write()
        wt.unlock()
        with tree.lock_tree_write():
            self.assertRaises(errors.PathsNotVersionedError, shelf.ShelfCreator, tree, tree.basis_tree(), ['foo'])

    def test_shelve_skips_added_root(self):
        """Skip adds of the root when iterating through shelvable changes."""
        tree = self.make_branch_and_tree('tree')
        tree.lock_tree_write()
        self.addCleanup(tree.unlock)
        creator = shelf.ShelfCreator(tree, tree.basis_tree())
        self.addCleanup(creator.finalize)
        self.assertEqual([], list(creator.iter_shelvable()))