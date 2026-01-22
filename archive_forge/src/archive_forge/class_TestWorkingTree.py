import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
class TestWorkingTree(TestCaseWithWorkingTree):

    def requireBranchReference(self):
        test_branch = self.make_branch('test-branch')
        try:
            test_branch.controldir.open_workingtree()
            raise TestNotApplicable('only on trees that can be separate from their branch.')
        except (errors.NoWorkingTree, errors.NotLocalUrl):
            pass

    def test_branch_builder(self):
        builder = self.make_branch_builder('foobar')
        br = _mod_branch.Branch.open(self.get_url('foobar'))

    def test_list_files(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['dir/', 'file'])
        if supports_symlinks(self.test_dir):
            os.symlink('target', 'symlink')
        tree.lock_read()
        files = list(tree.list_files())
        tree.unlock()
        self.assertEqual(files.pop(0), ('dir', '?', 'directory', TreeDirectory()))
        self.assertEqual(files.pop(0), ('file', '?', 'file', TreeFile()))
        if supports_symlinks(self.test_dir):
            self.assertEqual(files.pop(0), ('symlink', '?', 'symlink', TreeLink()))

    def test_list_files_sorted(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['dir/', 'file', 'dir/file', 'dir/b', 'dir/subdir/', 'a', 'dir/subfile', 'zz_dir/', 'zz_dir/subfile'])
        with tree.lock_read():
            files = [(path, kind) for path, v, kind, entry in tree.list_files()]
        self.assertEqual([('a', 'file'), ('dir', 'directory'), ('file', 'file'), ('zz_dir', 'directory')], files)
        with tree.lock_write():
            if tree.has_versioned_directories():
                tree.add(['dir', 'zz_dir'])
                files = [(path, kind) for path, v, kind, entry in tree.list_files()]
                self.assertEqual([('a', 'file'), ('dir', 'directory'), ('dir/b', 'file'), ('dir/file', 'file'), ('dir/subdir', 'directory'), ('dir/subfile', 'file'), ('file', 'file'), ('zz_dir', 'directory'), ('zz_dir/subfile', 'file')], files)
            else:
                tree.add(['dir/b'])
                files = [(path, kind) for path, v, kind, entry in tree.list_files()]
                self.assertEqual([('a', 'file'), ('dir', 'directory'), ('dir/b', 'file'), ('dir/file', 'file'), ('dir/subdir', 'directory'), ('dir/subfile', 'file'), ('file', 'file'), ('zz_dir', 'directory')], files)

    def test_transform(self):
        tree = self.make_branch_and_tree('tree')
        with tree.transform():
            pass

    def test_list_files_kind_change(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/filename'])
        tree.add('filename')
        os.unlink('tree/filename')
        self.build_tree(['tree/filename/'])
        tree.lock_read()
        self.addCleanup(tree.unlock)
        result = list(tree.list_files())
        self.assertEqual(1, len(result))
        if tree.has_versioned_directories():
            self.assertEqual(('filename', 'V', 'directory'), (result[0][0], result[0][1], result[0][2]))
        else:
            self.assertEqual(('filename', '?', 'directory'), (result[0][0], result[0][1], result[0][2]))

    def test_get_config_stack(self):
        wt = self.make_branch_and_tree('.')
        conf = wt.get_config_stack()
        self.assertIsInstance(conf, config.Stack)

    def test_open_containing(self):
        local_wt = self.make_branch_and_tree('.')
        local_url = local_wt.controldir.root_transport.base
        local_base = urlutils.local_path_from_url(local_url)
        del local_wt
        wt, relpath = WorkingTree.open_containing()
        self.assertEqual('', relpath)
        self.assertEqual(wt.basedir + '/', local_base)
        wt, relpath = WorkingTree.open_containing('.')
        self.assertEqual('', relpath)
        self.assertEqual(wt.basedir + '/', local_base)
        wt, relpath = WorkingTree.open_containing('./foo')
        self.assertEqual('foo', relpath)
        self.assertEqual(wt.basedir + '/', local_base)
        wt, relpath = WorkingTree.open_containing('./foo')
        wt, relpath = WorkingTree.open_containing(getcwd() + '/foo')
        self.assertEqual('foo', relpath)
        self.assertEqual(wt.basedir + '/', local_base)
        wt, relpath = WorkingTree.open_containing('./foo')
        wt, relpath = WorkingTree.open_containing(urlutils.local_path_to_url(getcwd() + '/foo'))
        self.assertEqual('foo', relpath)
        self.assertEqual(wt.basedir + '/', local_base)

    def test_basic_relpath(self):
        tree = self.make_branch_and_tree('.')
        self.assertEqual('child', tree.relpath(pathjoin(getcwd(), 'child')))

    def test_lock_locks_branch(self):
        tree = self.make_branch_and_tree('.')
        self.assertEqual(None, tree.branch.peek_lock_mode())
        with tree.lock_read():
            self.assertEqual('r', tree.branch.peek_lock_mode())
        self.assertEqual(None, tree.branch.peek_lock_mode())
        with tree.lock_write():
            self.assertEqual('w', tree.branch.peek_lock_mode())
        self.assertEqual(None, tree.branch.peek_lock_mode())

    def test_revert(self):
        """Test selected-file revert"""
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt'])
        with open('hello.txt', 'w') as f:
            f.write('initial hello')
        self.assertRaises(PathsNotVersionedError, tree.revert, ['hello.txt'])
        tree.add(['hello.txt'])
        tree.commit('create initial hello.txt')
        self.check_file_contents('hello.txt', b'initial hello')
        with open('hello.txt', 'w') as f:
            f.write('new hello')
        self.check_file_contents('hello.txt', b'new hello')
        tree.revert(['hello.txt'])
        self.check_file_contents('hello.txt', b'initial hello')
        self.check_file_contents('hello.txt.~1~', b'new hello')
        tree.revert(['hello.txt'])
        self.check_file_contents('hello.txt', b'initial hello')
        self.check_file_contents('hello.txt.~1~', b'new hello')
        with open('hello.txt', 'w') as f:
            f.write('new hello2')
        tree.revert(['hello.txt'])
        self.check_file_contents('hello.txt', b'initial hello')
        self.check_file_contents('hello.txt.~1~', b'new hello')
        self.check_file_contents('hello.txt.~2~', b'new hello2')

    def test_revert_missing(self):
        tree = self.make_branch_and_tree('.')
        with open('hello.txt', 'w') as f:
            f.write('initial hello')
        tree.add('hello.txt')
        tree.commit('added hello.txt')
        os.unlink('hello.txt')
        tree.remove('hello.txt')
        tree.revert(['hello.txt'])
        self.assertPathExists('hello.txt')

    def test_versioned_files_not_unknown(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt'])
        tree.add('hello.txt')
        self.assertEqual(list(tree.unknowns()), [])

    def test_unknowns(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['hello.txt', 'hello.txt.~1~'])
        self.build_tree_contents([('.bzrignore', b'*.~*\n')])
        tree.add('.bzrignore')
        self.assertEqual(list(tree.unknowns()), ['hello.txt'])

    def test_unknowns_empty_dir(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['subdir/', 'subdir/somefile'])
        if tree.has_versioned_directories():
            self.assertEqual(list(tree.unknowns()), ['subdir'])
        else:
            self.assertEqual(list(tree.unknowns()), ['subdir/somefile'])

    def test_initialize(self):
        t = self.make_branch_and_tree('.')
        b = _mod_branch.Branch.open('.')
        self.assertEqual(t.branch.base, b.base)
        t2 = WorkingTree.open('.')
        self.assertEqual(t.basedir, t2.basedir)
        self.assertEqual(b.base, t2.branch.base)

    def test_rename_dirs(self):
        """Test renaming directories and the files within them."""
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        self.build_tree(['dir/', 'dir/sub/', 'dir/sub/file'])
        wt.add(['dir', 'dir/sub', 'dir/sub/file'])
        wt.commit('create initial state')
        revid = b.last_revision()
        self.log('first revision_id is {%s}' % revid)
        tree = b.repository.revision_tree(revid)
        self.log('contents of tree: %r' % list(tree.iter_entries_by_dir()))
        self.check_tree_shape(tree, ['dir/', 'dir/sub/', 'dir/sub/file'])
        wt.rename_one('dir', 'newdir')
        wt.lock_read()
        self.check_tree_shape(wt, ['newdir/', 'newdir/sub/', 'newdir/sub/file'])
        wt.unlock()
        wt.rename_one('newdir/sub', 'newdir/newsub')
        wt.lock_read()
        self.check_tree_shape(wt, ['newdir/', 'newdir/newsub/', 'newdir/newsub/file'])
        wt.unlock()

    def test_add_in_unversioned(self):
        """Try to add a file in an unversioned directory.

        "bzr add" adds the parent as necessary, but simple working tree add
        doesn't do that.
        """
        from breezy.errors import NotVersionedError
        wt = self.make_branch_and_tree('.')
        self.build_tree(['foo/', 'foo/hello'])
        if not wt._format.supports_versioned_directories:
            wt.add('foo/hello')
        else:
            self.assertRaises(NotVersionedError, wt.add, 'foo/hello')

    def test_add_missing(self):
        wt = self.make_branch_and_tree('.')
        self.assertRaises(_mod_transport.NoSuchFile, wt.add, 'fpp')

    def test_remove_verbose(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['hello'])
        wt.add(['hello'])
        wt.commit(message='add hello')
        stdout = StringIO()
        stderr = StringIO()
        self.assertEqual(None, self.apply_redirected(None, stdout, stderr, wt.remove, ['hello'], verbose=True))
        self.assertEqual('?       hello\n', stdout.getvalue())
        self.assertEqual('', stderr.getvalue())

    def test_clone_trivial(self):
        wt = self.make_branch_and_tree('source')
        cloned_dir = wt.controldir.clone('target')
        cloned = cloned_dir.open_workingtree()
        self.assertEqual(cloned.get_parent_ids(), wt.get_parent_ids())

    def test_clone_empty(self):
        wt = self.make_branch_and_tree('source')
        cloned_dir = wt.controldir.clone('target', revision_id=_mod_revision.NULL_REVISION)
        cloned = cloned_dir.open_workingtree()
        self.assertEqual(cloned.get_parent_ids(), wt.get_parent_ids())

    def test_last_revision(self):
        wt = self.make_branch_and_tree('source')
        self.assertEqual([], wt.get_parent_ids())
        a = wt.commit('A', allow_pointless=True)
        parent_ids = wt.get_parent_ids()
        self.assertEqual([a], parent_ids)
        for parent_id in parent_ids:
            self.assertIsInstance(parent_id, bytes)

    def test_set_last_revision(self):
        wt = self.make_branch_and_tree('source')
        if wt.branch.repository._format.supports_ghosts:
            wt.set_last_revision(b'A')
        wt.set_last_revision(b'null:')
        a = wt.commit('A', allow_pointless=True)
        self.assertEqual([a], wt.get_parent_ids())
        wt.set_last_revision(b'null:')
        self.assertEqual([], wt.get_parent_ids())
        if getattr(wt.branch, '_set_revision_history', None) is None:
            raise TestSkipped('Branch format does not permit arbitrary history')
        wt.branch._set_revision_history([a, b'B'])
        wt.set_last_revision(a)
        self.assertEqual([a], wt.get_parent_ids())
        self.assertRaises(errors.ReservedId, wt.set_last_revision, b'A:')

    def test_set_last_revision_different_to_branch(self):
        self.requireBranchReference()
        wt = self.make_branch_and_tree('tree')
        a = wt.commit('A', allow_pointless=True)
        wt.set_last_revision(None)
        self.assertEqual([], wt.get_parent_ids())
        self.assertEqual(a, wt.branch.last_revision())
        wt.set_last_revision(a)
        self.assertEqual([a], wt.get_parent_ids())
        self.assertEqual(a, wt.branch.last_revision())

    def test_clone_and_commit_preserves_last_revision(self):
        """Doing a commit into a clone tree does not affect the source."""
        wt = self.make_branch_and_tree('source')
        cloned_dir = wt.controldir.clone('target')
        wt.commit('A', allow_pointless=True)
        self.assertNotEqual(cloned_dir.open_workingtree().get_parent_ids(), wt.get_parent_ids())

    def test_clone_preserves_content(self):
        wt = self.make_branch_and_tree('source')
        self.build_tree(['added', 'deleted', 'notadded'], transport=wt.controldir.transport.clone('..'))
        wt.add('deleted')
        wt.commit('add deleted')
        wt.remove('deleted')
        wt.add('added')
        cloned_dir = wt.controldir.clone('target')
        cloned = cloned_dir.open_workingtree()
        cloned_transport = cloned.controldir.transport.clone('..')
        self.assertFalse(cloned_transport.has('deleted'))
        self.assertTrue(cloned_transport.has('added'))
        self.assertFalse(cloned_transport.has('notadded'))
        self.assertTrue(cloned.is_versioned('added'))
        self.assertFalse(cloned.is_versioned('deleted'))
        self.assertFalse(cloned.is_versioned('notadded'))

    def test_basis_tree_returns_last_revision(self):
        wt = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        wt.add('foo')
        a = wt.commit('A')
        wt.rename_one('foo', 'bar')
        b = wt.commit('B')
        wt.set_parent_ids([b])
        tree = wt.basis_tree()
        tree.lock_read()
        self.assertTrue(tree.has_filename('bar'))
        tree.unlock()
        wt.set_parent_ids([a])
        tree = wt.basis_tree()
        tree.lock_read()
        self.assertTrue(tree.has_filename('foo'))
        tree.unlock()

    def test_clone_tree_revision(self):
        raise TestSkipped('revision limiting is not implemented yet.')

    def test_initialize_with_revision_id(self):
        source = self.make_branch_and_tree('source')
        a = source.commit('a', allow_pointless=True)
        source.commit('b', allow_pointless=True)
        self.build_tree(['new/'])
        made_control = self.bzrdir_format.initialize('new')
        source.branch.repository.clone(made_control)
        source.branch.clone(made_control)
        made_tree = self.workingtree_format.initialize(made_control, revision_id=a)
        self.assertEqual([a], made_tree.get_parent_ids())

    def test_post_build_tree_hook(self):
        calls = []

        def track_post_build_tree(tree):
            calls.append(tree.last_revision())
        source = self.make_branch_and_tree('source')
        a = source.commit('a', allow_pointless=True)
        source.commit('b', allow_pointless=True)
        self.build_tree(['new/'])
        made_control = self.bzrdir_format.initialize('new')
        source.branch.repository.clone(made_control)
        source.branch.clone(made_control)
        MutableTree.hooks.install_named_hook('post_build_tree', track_post_build_tree, 'Test')
        made_tree = self.workingtree_format.initialize(made_control, revision_id=a)
        self.assertEqual([a], calls)

    def test_update_sets_last_revision(self):
        self.requireBranchReference()
        wt = self.make_branch_and_tree('tree')
        self.build_tree(['checkout/', 'tree/file'])
        checkout = bzrdir.BzrDirMetaFormat1().initialize('checkout')
        checkout.set_branch_reference(wt.branch)
        old_tree = self.workingtree_format.initialize(checkout)
        wt.add('file')
        a = wt.commit('A')
        self.assertEqual(0, old_tree.update())
        self.assertPathExists('checkout/file')
        self.assertEqual([a], old_tree.get_parent_ids())

    def test_update_sets_root_id(self):
        """Ensure tree root is set properly by update.

        Since empty trees don't have root_ids, but workingtrees do,
        an update of a checkout of revision 0 to a new revision,  should set
        the root id.
        """
        wt = self.make_branch_and_tree('tree')
        main_branch = wt.branch
        self.build_tree(['checkout/', 'tree/file'])
        checkout = main_branch.create_checkout('checkout')
        wt.add('file')
        a = wt.commit('A')
        self.assertEqual(0, checkout.update())
        self.assertPathExists('checkout/file')
        if wt.supports_setting_file_ids():
            self.assertEqual(wt.path2id(''), checkout.path2id(''))
            self.assertNotEqual(None, wt.path2id(''))

    def test_update_sets_updated_root_id(self):
        wt = self.make_branch_and_tree('tree')
        if not wt.supports_setting_file_ids():
            self.assertRaises(SettingFileIdUnsupported, wt.set_root_id, 'first_root_id')
            return
        wt.set_root_id(b'first_root_id')
        self.assertEqual(b'first_root_id', wt.path2id(''))
        self.build_tree(['tree/file'])
        wt.add(['file'])
        wt.commit('first')
        co = wt.branch.create_checkout('checkout')
        wt.set_root_id(b'second_root_id')
        wt.commit('second')
        self.assertEqual(b'second_root_id', wt.path2id(''))
        self.assertEqual(0, co.update())
        self.assertEqual(b'second_root_id', co.path2id(''))

    def test_update_returns_conflict_count(self):
        self.requireBranchReference()
        wt = self.make_branch_and_tree('tree')
        self.build_tree(['checkout/', 'tree/file'])
        checkout = bzrdir.BzrDirMetaFormat1().initialize('checkout')
        checkout.set_branch_reference(wt.branch)
        old_tree = self.workingtree_format.initialize(checkout)
        wt.add('file')
        a = wt.commit('A')
        self.build_tree(['checkout/file'])
        old_tree.add('file')
        self.assertEqual(1, old_tree.update())
        self.assertEqual([a], old_tree.get_parent_ids())

    def test_merge_revert(self):
        from breezy.merge import merge_inner
        this = self.make_branch_and_tree('b1')
        self.build_tree_contents([('b1/a', b'a test\n'), ('b1/b', b'b test\n')])
        this.add(['a', 'b'])
        this.commit(message='')
        base = this.controldir.clone('b2').open_workingtree()
        self.build_tree_contents([('b2/a', b'b test\n')])
        other = this.controldir.clone('b3').open_workingtree()
        self.build_tree_contents([('b3/a', b'c test\n'), ('b3/c', b'c test\n')])
        other.add('c')
        self.build_tree_contents([('b1/b', b'q test\n'), ('b1/d', b'd test\n')])
        this.lock_write()
        self.addCleanup(this.unlock)
        merge_inner(this.branch, other, base, this_tree=this)
        with open('b1/a', 'rb') as a:
            self.assertNotEqual(a.read(), 'a test\n')
        this.revert()
        self.assertFileEqual(b'a test\n', 'b1/a')
        self.assertPathExists('b1/b.~1~')
        if this.supports_merge_modified():
            self.assertPathDoesNotExist('b1/c')
            self.assertPathDoesNotExist('b1/a.~1~')
        else:
            self.assertPathExists('b1/c')
            self.assertPathExists('b1/a.~1~')
        self.assertPathExists('b1/d')

    def test_update_updates_bound_branch_no_local_commits(self):
        master_tree = self.make_branch_and_tree('master')
        tree = self.make_branch_and_tree('tree')
        try:
            tree.branch.bind(master_tree.branch)
        except _mod_branch.BindingUnsupported:
            return
        foo = master_tree.commit('foo', allow_pointless=True)
        tree.update()
        self.assertEqual([foo], tree.get_parent_ids())
        self.assertEqual(foo, tree.branch.last_revision())

    def test_update_turns_local_commit_into_merge(self):
        master_tree = self.make_branch_and_tree('master')
        master_tip = master_tree.commit('first master commit')
        tree = self.make_branch_and_tree('tree')
        try:
            tree.branch.bind(master_tree.branch)
        except _mod_branch.BindingUnsupported:
            return
        tree.update()
        tree.commit('foo', allow_pointless=True, local=True)
        bar = tree.commit('bar', allow_pointless=True, local=True)
        tree.update()
        self.assertEqual([master_tip, bar], tree.get_parent_ids())
        self.assertEqual(master_tree.branch.last_revision(), tree.branch.last_revision())

    def test_update_takes_revision_parameter(self):
        wt = self.make_branch_and_tree('wt')
        self.build_tree_contents([('wt/a', b'old content')])
        wt.add(['a'])
        rev1 = wt.commit('first master commit')
        self.build_tree_contents([('wt/a', b'new content')])
        rev2 = wt.commit('second master commit')
        conflicts = wt.update(revision=rev1)
        self.assertFileEqual(b'old content', 'wt/a')
        self.assertEqual([rev1], wt.get_parent_ids())

    def test_merge_modified_detects_corruption(self):
        tree = self.make_branch_and_tree('master')
        if not isinstance(tree, InventoryWorkingTree):
            raise TestNotApplicable('merge-hashes is specific to bzr working trees')
        tree._transport.put_bytes('merge-hashes', b'asdfasdf')
        self.assertRaises(errors.MergeModifiedFormatError, tree.merge_modified)

    def test_merge_modified(self):
        tree = self.make_branch_and_tree('tree')
        self.build_tree_contents([('tree/somefile', b'hello')])
        with tree.lock_write():
            tree.add(['somefile'])
            d = {'somefile': osutils.sha_string(b'hello')}
            if tree.supports_merge_modified():
                tree.set_merge_modified(d)
                mm = tree.merge_modified()
                self.assertEqual(mm, d)
            else:
                self.assertRaises(errors.UnsupportedOperation, tree.set_merge_modified, d)
                mm = tree.merge_modified()
                self.assertEqual(mm, {})
        if tree.supports_merge_modified():
            mm = tree.merge_modified()
            self.assertEqual(mm, d)
        else:
            mm = tree.merge_modified()
            self.assertEqual(mm, {})

    def test_conflicts(self):
        from breezy.tests.test_conflicts import example_conflicts
        tree = self.make_branch_and_tree('master')
        try:
            tree.set_conflicts(example_conflicts)
        except UnsupportedOperation:
            raise TestSkipped('set_conflicts not supported')
        tree2 = WorkingTree.open('master')
        self.assertEqual(tree2.conflicts(), example_conflicts)
        tree2._transport.put_bytes('conflicts', b'')
        self.assertRaises(errors.ConflictFormatError, tree2.conflicts)
        tree2._transport.put_bytes('conflicts', b'a')
        self.assertRaises(errors.ConflictFormatError, tree2.conflicts)

    def make_merge_conflicts(self):
        from breezy.merge import merge_inner
        tree = self.make_branch_and_tree('mine')
        with open('mine/bloo', 'wb') as f:
            f.write(b'one')
        with open('mine/blo', 'wb') as f:
            f.write(b'on')
        tree.add(['bloo', 'blo'])
        tree.commit('blah', allow_pointless=False)
        base = tree.branch.repository.revision_tree(tree.last_revision())
        controldir.ControlDir.open('mine').sprout('other')
        with open('other/bloo', 'wb') as f:
            f.write(b'two')
        othertree = WorkingTree.open('other')
        othertree.commit('blah', allow_pointless=False)
        with open('mine/bloo', 'wb') as f:
            f.write(b'three')
        tree.commit('blah', allow_pointless=False)
        merge_inner(tree.branch, othertree, base, this_tree=tree)
        return tree

    def test_merge_conflicts(self):
        tree = self.make_merge_conflicts()
        self.assertEqual(len(tree.conflicts()), 1)

    def test_clear_merge_conflicts(self):
        tree = self.make_merge_conflicts()
        self.assertEqual(len(tree.conflicts()), 1)
        try:
            tree.set_conflicts([])
        except UnsupportedOperation:
            raise TestSkipped('unsupported operation')
        self.assertEqual(tree.conflicts(), ConflictList())

    def test_add_conflicts(self):
        tree = self.make_branch_and_tree('tree')
        try:
            tree.add_conflicts([TextConflict('path_a')])
        except UnsupportedOperation:
            raise TestSkipped('unsupported operation')
        self.assertEqual(ConflictList([TextConflict('path_a')]), tree.conflicts())
        tree.add_conflicts([TextConflict('path_a')])
        self.assertEqual(ConflictList([TextConflict('path_a')]), tree.conflicts())
        tree.add_conflicts([ContentsConflict('path_a')])
        self.assertEqual(ConflictList([ContentsConflict('path_a'), TextConflict('path_a')]), tree.conflicts())
        tree.add_conflicts([TextConflict('path_b')])
        self.assertEqual(ConflictList([ContentsConflict('path_a'), TextConflict('path_a'), TextConflict('path_b')]), tree.conflicts())

    def test_revert_clear_conflicts(self):
        tree = self.make_merge_conflicts()
        self.assertEqual(len(tree.conflicts()), 1)
        tree.revert(['blo'])
        self.assertEqual(len(tree.conflicts()), 1)
        tree.revert(['bloo'])
        self.assertEqual(len(tree.conflicts()), 0)

    def test_revert_clear_conflicts2(self):
        tree = self.make_merge_conflicts()
        self.assertEqual(len(tree.conflicts()), 1)
        tree.revert()
        self.assertEqual(len(tree.conflicts()), 0)

    def test_format_description(self):
        tree = self.make_branch_and_tree('tree')
        text = tree._format.get_format_description()
        self.assertTrue(len(text))

    def test_format_leftmost_parent_id_as_ghost(self):
        tree = self.make_branch_and_tree('tree')
        self.assertIn(tree._format.supports_leftmost_parent_id_as_ghost, (True, False))

    def test_branch_attribute_is_not_settable(self):
        tree = self.make_branch_and_tree('tree')

        def set_branch():
            tree.branch = tree.branch
        self.assertRaises(AttributeError, set_branch)

    def test_list_files_versioned_before_ignored(self):
        """A versioned file matching an ignore rule should not be ignored."""
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo.pyc'])
        self.build_tree_contents([('.bzrignore', b'foo.pyc')])
        tree.add('foo.pyc')
        tree.lock_read()
        files = sorted(list(tree.list_files()))
        tree.unlock()
        self.assertEqual(('.bzrignore', '?', 'file', None), (files[0][0], files[0][1], files[0][2], getattr(files[0][3], 'file_id', None)))
        self.assertEqual(('foo.pyc', 'V', 'file'), (files[1][0], files[1][1], files[1][2]))
        self.assertEqual(2, len(files))

    def test_non_normalized_add_accessible(self):
        try:
            self.build_tree(['å'])
        except UnicodeError:
            raise TestSkipped('Filesystem does not support unicode filenames')
        tree = self.make_branch_and_tree('.')
        orig = osutils.normalized_filename
        osutils.normalized_filename = osutils._accessible_normalized_filename
        try:
            tree.add(['å'])
            with tree.lock_read():
                self.assertEqual([('', 'directory'), ('å', 'file')], [(path, ie.kind) for path, ie in tree.iter_entries_by_dir()])
        finally:
            osutils.normalized_filename = orig

    def test_non_normalized_add_inaccessible(self):
        try:
            self.build_tree(['å'])
        except UnicodeError:
            raise TestSkipped('Filesystem does not support unicode filenames')
        tree = self.make_branch_and_tree('.')
        orig = osutils.normalized_filename
        osutils.normalized_filename = osutils._inaccessible_normalized_filename
        try:
            self.assertRaises(errors.InvalidNormalization, tree.add, ['å'])
        finally:
            osutils.normalized_filename = orig

    def test__write_inventory(self):
        tree = self.make_branch_and_tree('.')
        if not isinstance(tree, InventoryWorkingTree):
            raise TestNotApplicable('_write_inventory does not exist on non-inventory working trees')
        self.build_tree(['present', 'unknown'])
        inventory = Inventory(tree.path2id(''))
        inventory.add_path('missing', 'file', b'missing-id')
        inventory.add_path('present', 'file', b'present-id')
        tree.lock_write()
        tree._write_inventory(inventory)
        tree.unlock()
        with tree.lock_read():
            present_stat = os.lstat('present')
            unknown_stat = os.lstat('unknown')
            expected_results = [('', [('missing', 'missing', 'unknown', None, 'file'), ('present', 'present', 'file', present_stat, 'file'), ('unknown', 'unknown', 'file', unknown_stat, None)])]
            self.assertEqual(expected_results, list(tree.walkdirs()))

    def test_path2id(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['foo'])
        if tree.supports_setting_file_ids():
            tree.add(['foo'], ids=[b'foo-id'])
            self.assertEqual(b'foo-id', tree.path2id('foo'))
            self.assertEqual(b'foo-id', tree.path2id('foo/'))
        else:
            tree.add(['foo'])
            if tree.branch.repository._format.supports_versioned_directories:
                self.assertIsInstance(str, tree.path2id('foo'))
            else:
                self.skipTest('format does not support versioning directories')

    def test_filter_unversioned_files(self):
        tree = self.make_branch_and_tree('.')
        paths = ['here-and-versioned', 'here-and-not-versioned', 'not-here-and-versioned', 'not-here-and-not-versioned']
        tree.add(['here-and-versioned', 'not-here-and-versioned'], kinds=['file', 'file'])
        self.build_tree(['here-and-versioned', 'here-and-not-versioned'])
        tree.lock_read()
        self.addCleanup(tree.unlock)
        self.assertEqual({'not-here-and-not-versioned', 'here-and-not-versioned'}, tree.filter_unversioned_files(paths))

    def test_detect_real_kind(self):
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['file', 'directory/'])
        names = ['file', 'directory']
        if supports_symlinks(self.test_dir):
            os.symlink('target', 'symlink')
            names.append('symlink')
        tree.add(names)
        for n in names:
            actual_kind = tree.kind(n)
            self.assertEqual(n, actual_kind)
        os.rename(names[0], 'tmp')
        for i in range(1, len(names)):
            os.rename(names[i], names[i - 1])
        os.rename('tmp', names[-1])
        for i in range(len(names)):
            actual_kind = tree.kind(names[i - 1])
            expected_kind = names[i]
            self.assertEqual(expected_kind, actual_kind)

    def test_stored_kind_with_missing(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/a', 'tree/b/'])
        tree.add(['a', 'b'])
        os.unlink('tree/a')
        os.rmdir('tree/b')
        self.assertEqual('file', tree.stored_kind('a'))
        if tree.branch.repository._format.supports_versioned_directories:
            self.assertEqual('directory', tree.stored_kind('b'))

    def test_stored_kind_nonexistent(self):
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.assertRaises(_mod_transport.NoSuchFile, tree.stored_kind, 'a')
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/a'])
        self.assertRaises(_mod_transport.NoSuchFile, tree.stored_kind, 'a')
        tree.add(['a'])
        self.assertIs('file', tree.stored_kind('a'))

    def test_missing_file_sha1(self):
        """If a file is missing, its sha1 should be reported as None."""
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['file'])
        tree.add('file')
        tree.commit('file added')
        os.unlink('file')
        self.assertIs(None, tree.get_file_sha1('file'))

    def test_no_file_sha1(self):
        """If a file is not present, get_file_sha1 should raise NoSuchFile"""
        tree = self.make_branch_and_tree('.')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.assertRaises(_mod_transport.NoSuchFile, tree.get_file_sha1, 'nonexistant')
        self.build_tree(['file'])
        tree.add('file')
        tree.commit('foo')
        tree.remove('file')
        self.assertRaises(_mod_transport.NoSuchFile, tree.get_file_sha1, 'file')

    def test_case_sensitive(self):
        """If filesystem is case-sensitive, tree should report this.

        We check case-sensitivity by creating a file with a lowercase name,
        then testing whether it exists with an uppercase name.
        """
        self.build_tree(['filename'])
        case_sensitive = not features.CaseInsensitiveFilesystemFeature.available()
        tree = self.make_branch_and_tree('test')
        self.assertEqual(case_sensitive, tree.case_sensitive)
        if not isinstance(tree, InventoryWorkingTree):
            raise TestNotApplicable('get_format_string is only available on bzr working trees')
        t = tree.controldir.get_workingtree_transport(None)
        try:
            content = tree._format.get_format_string()
        except NotImplementedError:
            content = tree.controldir._format.get_format_string()
        t.put_bytes(tree._format.case_sensitive_filename, content)
        tree = tree.controldir.open_workingtree()
        self.assertFalse(tree.case_sensitive)

    def test_supports_executable(self):
        self.build_tree(['filename'])
        tree = self.make_branch_and_tree('.')
        tree.add('filename')
        self.assertIsInstance(tree._supports_executable(), bool)
        if tree._supports_executable():
            tree.lock_read()
            try:
                self.assertFalse(tree.is_executable('filename'))
            finally:
                tree.unlock()
            os.chmod('filename', 493)
            self.addCleanup(tree.lock_read().unlock)
            self.assertTrue(tree.is_executable('filename'))
        else:
            self.addCleanup(tree.lock_read().unlock)
            self.assertFalse(tree.is_executable('filename'))

    def test_all_file_ids_with_missing(self):
        if not self.workingtree_format.supports_setting_file_ids:
            raise TestNotApplicable('does not support setting file ids')
        tree = self.make_branch_and_tree('tree')
        tree.lock_write()
        self.addCleanup(tree.unlock)
        self.build_tree(['tree/a', 'tree/b'])
        tree.add(['a', 'b'])
        os.unlink('tree/a')
        self.assertEqual({'a', 'b', ''}, set(tree.all_versioned_paths()))

    def test_sprout_hardlink(self):
        real_os_link = getattr(os, 'link', None)
        if real_os_link is None:
            raise TestNotApplicable("This platform doesn't provide os.link")
        source = self.make_branch_and_tree('source')
        self.build_tree(['source/file'])
        source.add('file')
        source.commit('added file')

        def fake_link(source, target):
            raise OSError(errno.EPERM, 'Operation not permitted')
        os.link = fake_link
        try:
            try:
                source.controldir.sprout('target', accelerator_tree=source, hardlink=True)
            except errors.HardLinkNotSupported:
                pass
        finally:
            os.link = real_os_link