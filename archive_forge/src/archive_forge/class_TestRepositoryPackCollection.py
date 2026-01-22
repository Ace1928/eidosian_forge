from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
class TestRepositoryPackCollection(TestCaseWithTransport):

    def get_format(self):
        return controldir.format_registry.make_controldir('pack-0.92')

    def get_packs(self):
        format = self.get_format()
        repo = self.make_repository('.', format=format)
        return repo._pack_collection

    def make_packs_and_alt_repo(self, write_lock=False):
        """Create a pack repo with 3 packs, and access it via a second repo."""
        tree = self.make_branch_and_tree('.', format=self.get_format())
        tree.lock_write()
        self.addCleanup(tree.unlock)
        rev1 = tree.commit('one')
        rev2 = tree.commit('two')
        rev3 = tree.commit('three')
        r = repository.Repository.open('.')
        if write_lock:
            r.lock_write()
        else:
            r.lock_read()
        self.addCleanup(r.unlock)
        packs = r._pack_collection
        packs.ensure_loaded()
        return (tree, r, packs, [rev1, rev2, rev3])

    def test__clear_obsolete_packs(self):
        packs = self.get_packs()
        obsolete_pack_trans = packs.transport.clone('obsolete_packs')
        obsolete_pack_trans.put_bytes('a-pack.pack', b'content\n')
        obsolete_pack_trans.put_bytes('a-pack.rix', b'content\n')
        obsolete_pack_trans.put_bytes('a-pack.iix', b'content\n')
        obsolete_pack_trans.put_bytes('another-pack.pack', b'foo\n')
        obsolete_pack_trans.put_bytes('not-a-pack.rix', b'foo\n')
        res = packs._clear_obsolete_packs()
        self.assertEqual(['a-pack', 'another-pack'], sorted(res))
        self.assertEqual([], obsolete_pack_trans.list_dir('.'))

    def test__clear_obsolete_packs_preserve(self):
        packs = self.get_packs()
        obsolete_pack_trans = packs.transport.clone('obsolete_packs')
        obsolete_pack_trans.put_bytes('a-pack.pack', b'content\n')
        obsolete_pack_trans.put_bytes('a-pack.rix', b'content\n')
        obsolete_pack_trans.put_bytes('a-pack.iix', b'content\n')
        obsolete_pack_trans.put_bytes('another-pack.pack', b'foo\n')
        obsolete_pack_trans.put_bytes('not-a-pack.rix', b'foo\n')
        res = packs._clear_obsolete_packs(preserve={'a-pack'})
        self.assertEqual(['a-pack', 'another-pack'], sorted(res))
        self.assertEqual(['a-pack.iix', 'a-pack.pack', 'a-pack.rix'], sorted(obsolete_pack_trans.list_dir('.')))

    def test__max_pack_count(self):
        """The maximum pack count is a function of the number of revisions."""
        packs = self.get_packs()
        self.assertEqual(1, packs._max_pack_count(0))
        self.assertEqual(1, packs._max_pack_count(1))
        self.assertEqual(2, packs._max_pack_count(2))
        self.assertEqual(3, packs._max_pack_count(3))
        self.assertEqual(4, packs._max_pack_count(4))
        self.assertEqual(5, packs._max_pack_count(5))
        self.assertEqual(6, packs._max_pack_count(6))
        self.assertEqual(7, packs._max_pack_count(7))
        self.assertEqual(8, packs._max_pack_count(8))
        self.assertEqual(9, packs._max_pack_count(9))
        self.assertEqual(1, packs._max_pack_count(10))
        self.assertEqual(2, packs._max_pack_count(11))
        self.assertEqual(10, packs._max_pack_count(19))
        self.assertEqual(2, packs._max_pack_count(20))
        self.assertEqual(3, packs._max_pack_count(21))
        self.assertEqual(25, packs._max_pack_count(112894))

    def test_repr(self):
        packs = self.get_packs()
        self.assertContainsRe(repr(packs), 'RepositoryPackCollection(.*Repository(.*))')

    def test__obsolete_packs(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        names = packs.names()
        pack = packs.get_pack_by_name(names[0])
        packs._remove_pack_from_memory(pack)
        packs.transport.rename('packs/{}.pack'.format(names[0]), 'obsolete_packs/{}.pack'.format(names[0]))
        packs.transport.rename('indices/{}.iix'.format(names[0]), 'obsolete_packs/{}.iix'.format(names[0]))
        packs._obsolete_packs([pack])
        self.assertEqual([n + '.pack' for n in names[1:]], sorted(packs._pack_transport.list_dir('.')))
        self.assertEqual(names[1:], sorted({osutils.splitext(n)[0] for n in packs._index_transport.list_dir('.')}))

    def test__obsolete_packs_missing_directory(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        r.control_transport.rmdir('obsolete_packs')
        names = packs.names()
        pack = packs.get_pack_by_name(names[0])
        packs._remove_pack_from_memory(pack)
        packs._obsolete_packs([pack])
        self.assertEqual([n + '.pack' for n in names[1:]], sorted(packs._pack_transport.list_dir('.')))
        self.assertEqual(names[1:], sorted({osutils.splitext(n)[0] for n in packs._index_transport.list_dir('.')}))

    def test_pack_distribution_zero(self):
        packs = self.get_packs()
        self.assertEqual([0], packs.pack_distribution(0))

    def test_ensure_loaded_unlocked(self):
        packs = self.get_packs()
        self.assertRaises(errors.ObjectNotLocked, packs.ensure_loaded)

    def test_pack_distribution_one_to_nine(self):
        packs = self.get_packs()
        self.assertEqual([1], packs.pack_distribution(1))
        self.assertEqual([1, 1], packs.pack_distribution(2))
        self.assertEqual([1, 1, 1], packs.pack_distribution(3))
        self.assertEqual([1, 1, 1, 1], packs.pack_distribution(4))
        self.assertEqual([1, 1, 1, 1, 1], packs.pack_distribution(5))
        self.assertEqual([1, 1, 1, 1, 1, 1], packs.pack_distribution(6))
        self.assertEqual([1, 1, 1, 1, 1, 1, 1], packs.pack_distribution(7))
        self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1], packs.pack_distribution(8))
        self.assertEqual([1, 1, 1, 1, 1, 1, 1, 1, 1], packs.pack_distribution(9))

    def test_pack_distribution_stable_at_boundaries(self):
        """When there are multi-rev packs the counts are stable."""
        packs = self.get_packs()
        self.assertEqual([10], packs.pack_distribution(10))
        self.assertEqual([10, 1], packs.pack_distribution(11))
        self.assertEqual([10, 10], packs.pack_distribution(20))
        self.assertEqual([10, 10, 1], packs.pack_distribution(21))
        self.assertEqual([100], packs.pack_distribution(100))
        self.assertEqual([100, 1], packs.pack_distribution(101))
        self.assertEqual([100, 10, 1], packs.pack_distribution(111))
        self.assertEqual([100, 100], packs.pack_distribution(200))
        self.assertEqual([100, 100, 1], packs.pack_distribution(201))
        self.assertEqual([100, 100, 10, 1], packs.pack_distribution(211))

    def test_plan_pack_operations_2009_revisions_skip_all_packs(self):
        packs = self.get_packs()
        existing_packs = [(2000, 'big'), (9, 'medium')]
        pack_operations = packs.plan_autopack_combinations(existing_packs, [1000, 1000, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([], pack_operations)

    def test_plan_pack_operations_2010_revisions_skip_all_packs(self):
        packs = self.get_packs()
        existing_packs = [(2000, 'big'), (9, 'medium'), (1, 'single')]
        pack_operations = packs.plan_autopack_combinations(existing_packs, [1000, 1000, 10])
        self.assertEqual([], pack_operations)

    def test_plan_pack_operations_2010_combines_smallest_two(self):
        packs = self.get_packs()
        existing_packs = [(1999, 'big'), (9, 'medium'), (1, 'single2'), (1, 'single1')]
        pack_operations = packs.plan_autopack_combinations(existing_packs, [1000, 1000, 10])
        self.assertEqual([[2, ['single2', 'single1']]], pack_operations)

    def test_plan_pack_operations_creates_a_single_op(self):
        packs = self.get_packs()
        existing_packs = [(50, 'a'), (40, 'b'), (30, 'c'), (10, 'd'), (10, 'e'), (6, 'f'), (4, 'g')]
        distribution = packs.pack_distribution(150)
        pack_operations = packs.plan_autopack_combinations(existing_packs, distribution)
        self.assertEqual([[130, ['a', 'b', 'c', 'f', 'g']]], pack_operations)

    def test_all_packs_none(self):
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        tree.lock_read()
        self.addCleanup(tree.unlock)
        packs = tree.branch.repository._pack_collection
        packs.ensure_loaded()
        self.assertEqual([], packs.all_packs())

    def test_all_packs_one(self):
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        tree.commit('start')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        packs = tree.branch.repository._pack_collection
        packs.ensure_loaded()
        self.assertEqual([packs.get_pack_by_name(packs.names()[0])], packs.all_packs())

    def test_all_packs_two(self):
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        tree.commit('start')
        tree.commit('continue')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        packs = tree.branch.repository._pack_collection
        packs.ensure_loaded()
        self.assertEqual([packs.get_pack_by_name(packs.names()[0]), packs.get_pack_by_name(packs.names()[1])], packs.all_packs())

    def test_get_pack_by_name(self):
        format = self.get_format()
        tree = self.make_branch_and_tree('.', format=format)
        tree.commit('start')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        packs = tree.branch.repository._pack_collection
        packs.reset()
        packs.ensure_loaded()
        name = packs.names()[0]
        pack_1 = packs.get_pack_by_name(name)
        sizes = packs._names[name]
        rev_index = GraphIndex(packs._index_transport, name + '.rix', sizes[0])
        inv_index = GraphIndex(packs._index_transport, name + '.iix', sizes[1])
        txt_index = GraphIndex(packs._index_transport, name + '.tix', sizes[2])
        sig_index = GraphIndex(packs._index_transport, name + '.six', sizes[3])
        self.assertEqual(pack_repo.ExistingPack(packs._pack_transport, name, rev_index, inv_index, txt_index, sig_index), pack_1)
        self.assertTrue(pack_1 is packs.get_pack_by_name(name))

    def test_reload_pack_names_new_entry(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo()
        names = packs.names()
        rev4 = tree.commit('four')
        new_names = tree.branch.repository._pack_collection.names()
        new_name = set(new_names).difference(names)
        self.assertEqual(1, len(new_name))
        new_name = new_name.pop()
        self.assertEqual(names, packs.names())
        self.assertTrue(packs.reload_pack_names())
        self.assertEqual(new_names, packs.names())
        self.assertEqual({rev4: (revs[-1],)}, r.get_parent_map([rev4]))
        self.assertFalse(packs.reload_pack_names())

    def test_reload_pack_names_added_and_removed(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo()
        names = packs.names()
        tree.branch.repository.pack()
        new_names = tree.branch.repository._pack_collection.names()
        self.assertEqual(names, packs.names())
        self.assertTrue(packs.reload_pack_names())
        self.assertEqual(new_names, packs.names())
        self.assertEqual({revs[-1]: (revs[-2],)}, r.get_parent_map([revs[-1]]))
        self.assertFalse(packs.reload_pack_names())

    def test_reload_pack_names_preserves_pending(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        orig_names = packs.names()
        orig_at_load = packs._packs_at_load
        to_remove_name = next(iter(orig_names))
        r.start_write_group()
        self.addCleanup(r.abort_write_group)
        r.texts.insert_record_stream([versionedfile.FulltextContentFactory((b'text', b'rev'), (), None, b'content\n')])
        new_pack = packs._new_pack
        self.assertTrue(new_pack.data_inserted())
        new_pack.finish()
        packs.allocate(new_pack)
        packs._new_pack = None
        removed_pack = packs.get_pack_by_name(to_remove_name)
        packs._remove_pack_from_memory(removed_pack)
        names = packs.names()
        all_nodes, deleted_nodes, new_nodes, _ = packs._diff_pack_names()
        new_names = {x[0] for x in new_nodes}
        self.assertEqual(names, sorted([x[0] for x in all_nodes]))
        self.assertEqual(set(names) - set(orig_names), new_names)
        self.assertEqual({new_pack.name}, new_names)
        self.assertEqual([to_remove_name], sorted([x[0] for x in deleted_nodes]))
        packs.reload_pack_names()
        reloaded_names = packs.names()
        self.assertEqual(orig_at_load, packs._packs_at_load)
        self.assertEqual(names, reloaded_names)
        all_nodes, deleted_nodes, new_nodes, _ = packs._diff_pack_names()
        new_names = {x[0] for x in new_nodes}
        self.assertEqual(names, sorted([x[0] for x in all_nodes]))
        self.assertEqual(set(names) - set(orig_names), new_names)
        self.assertEqual({new_pack.name}, new_names)
        self.assertEqual([to_remove_name], sorted([x[0] for x in deleted_nodes]))

    def test_autopack_obsoletes_new_pack(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        packs._max_pack_count = lambda x: 1
        packs.pack_distribution = lambda x: [10]
        r.start_write_group()
        r.revisions.insert_record_stream([versionedfile.FulltextContentFactory((b'bogus-rev',), (), None, b'bogus-content\n')])
        r.commit_write_group()
        names = packs.names()
        self.assertEqual(1, len(names))
        self.assertEqual([names[0] + '.pack'], packs._pack_transport.list_dir('.'))

    def test_autopack_reloads_and_stops(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        orig_execute = packs._execute_pack_operations

        def _munged_execute_pack_ops(*args, **kwargs):
            tree.branch.repository.pack()
            return orig_execute(*args, **kwargs)
        packs._execute_pack_operations = _munged_execute_pack_ops
        packs._max_pack_count = lambda x: 1
        packs.pack_distribution = lambda x: [10]
        self.assertFalse(packs.autopack())
        self.assertEqual(1, len(packs.names()))
        self.assertEqual(tree.branch.repository._pack_collection.names(), packs.names())

    def test__save_pack_names(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        names = packs.names()
        pack = packs.get_pack_by_name(names[0])
        packs._remove_pack_from_memory(pack)
        packs._save_pack_names(obsolete_packs=[pack])
        cur_packs = packs._pack_transport.list_dir('.')
        self.assertEqual([n + '.pack' for n in names[1:]], sorted(cur_packs))
        obsolete_packs = packs.transport.list_dir('obsolete_packs')
        obsolete_names = {osutils.splitext(n)[0] for n in obsolete_packs}
        self.assertEqual([pack.name], sorted(obsolete_names))

    def test__save_pack_names_already_obsoleted(self):
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        names = packs.names()
        pack = packs.get_pack_by_name(names[0])
        packs._remove_pack_from_memory(pack)
        packs._obsolete_packs([pack])
        packs._save_pack_names(clear_obsolete_packs=True, obsolete_packs=[pack])
        cur_packs = packs._pack_transport.list_dir('.')
        self.assertEqual([n + '.pack' for n in names[1:]], sorted(cur_packs))
        obsolete_packs = packs.transport.list_dir('obsolete_packs')
        obsolete_names = {osutils.splitext(n)[0] for n in obsolete_packs}
        self.assertEqual([pack.name], sorted(obsolete_names))

    def test_pack_no_obsolete_packs_directory(self):
        """Bug #314314, don't fail if obsolete_packs directory does
        not exist."""
        tree, r, packs, revs = self.make_packs_and_alt_repo(write_lock=True)
        r.control_transport.rmdir('obsolete_packs')
        packs._clear_obsolete_packs()