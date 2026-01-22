from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestCHKMapDifference(TestCaseWithExampleMaps):

    def get_difference(self, new_roots, old_roots, search_key_func=None):
        if search_key_func is None:
            search_key_func = chk_map._search_key_plain
        return chk_map.CHKMapDifference(self.get_chk_bytes(), new_roots, old_roots, search_key_func)

    def test__init__(self):
        c_map = self.make_root_only_map()
        key1 = c_map.key()
        c_map.map((b'aaa',), b'new aaa content')
        key2 = c_map._save()
        diff = self.get_difference([key2], [key1])
        self.assertEqual({key1}, diff._all_old_chks)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([], diff._new_queue)

    def help__read_all_roots(self, search_key_func):
        c_map = self.make_root_only_map(search_key_func=search_key_func)
        key1 = c_map.key()
        c_map.map((b'aaa',), b'new aaa content')
        key2 = c_map._save()
        diff = self.get_difference([key2], [key1], search_key_func)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([((b'aaa',), b'new aaa content')], diff._new_item_queue)
        self.assertEqual([], diff._new_queue)
        self.assertEqual([], diff._old_queue)

    def test__read_all_roots_plain(self):
        self.help__read_all_roots(search_key_func=chk_map._search_key_plain)

    def test__read_all_roots_16(self):
        self.help__read_all_roots(search_key_func=chk_map._search_key_16)

    def test__read_all_roots_skips_known_old(self):
        c_map = self.make_one_deep_map(chk_map._search_key_plain)
        key1 = c_map.key()
        c_map2 = self.make_root_only_map(chk_map._search_key_plain)
        key2 = c_map2.key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)

    def test__read_all_roots_prepares_queues(self):
        c_map = self.make_one_deep_map(chk_map._search_key_plain)
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items[b'a'].key()
        c_map.map((b'abb',), b'new abb content')
        key2 = c_map._save()
        key2_a = c_map._root_node._items[b'a'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key2_a], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        self.assertEqual([key1_a], diff._old_queue)

    def test__read_all_roots_multi_new_prepares_queues(self):
        c_map = self.make_one_deep_map(chk_map._search_key_plain)
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items[b'a'].key()
        key1_c = c_map._root_node._items[b'c'].key()
        c_map.map((b'abb',), b'new abb content')
        key2 = c_map._save()
        key2_a = c_map._root_node._items[b'a'].key()
        key2_c = c_map._root_node._items[b'c'].key()
        c_map = chk_map.CHKMap(self.get_chk_bytes(), key1, chk_map._search_key_plain)
        c_map.map((b'ccc',), b'new ccc content')
        key3 = c_map._save()
        key3_a = c_map._root_node._items[b'a'].key()
        key3_c = c_map._root_node._items[b'c'].key()
        diff = self.get_difference([key2, key3], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual(sorted([key2, key3]), sorted(root_results))
        self.assertEqual({key2_a, key3_c}, set(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)
        self.assertEqual({key1_a, key1_c}, set(diff._old_queue))

    def test__read_all_roots_different_depths(self):
        c_map = self.make_two_deep_map(chk_map._search_key_plain)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_a = c_map._root_node._items[b'a'].key()
        key1_c = c_map._root_node._items[b'c'].key()
        key1_d = c_map._root_node._items[b'd'].key()
        c_map2 = self.make_one_deep_two_prefix_map(chk_map._search_key_plain)
        c_map2._dump_tree()
        key2 = c_map2.key()
        key2_aa = c_map2._root_node._items[b'aa'].key()
        key2_ad = c_map2._root_node._items[b'ad'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key1_a], diff._old_queue)
        self.assertEqual({key2_aa, key2_ad}, set(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)
        diff = self.get_difference([key1], [key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key1], root_results)
        self.assertEqual({key2_aa, key2_ad}, set(diff._old_queue))
        self.assertEqual({key1_a, key1_c, key1_d}, set(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_different_depths_16(self):
        c_map = self.make_two_deep_map(chk_map._search_key_16)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_2 = c_map._root_node._items[b'2'].key()
        key1_4 = c_map._root_node._items[b'4'].key()
        key1_C = c_map._root_node._items[b'C'].key()
        key1_F = c_map._root_node._items[b'F'].key()
        c_map2 = self.make_one_deep_two_prefix_map(chk_map._search_key_16)
        c_map2._dump_tree()
        key2 = c_map2.key()
        key2_F0 = c_map2._root_node._items[b'F0'].key()
        key2_F3 = c_map2._root_node._items[b'F3'].key()
        key2_F4 = c_map2._root_node._items[b'F4'].key()
        key2_FD = c_map2._root_node._items[b'FD'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key1_F], diff._old_queue)
        self.assertEqual(sorted([key2_F0, key2_F3, key2_F4, key2_FD]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)
        diff = self.get_difference([key1], [key2], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key1], root_results)
        self.assertEqual(sorted([key2_F0, key2_F3, key2_F4, key2_FD]), sorted(diff._old_queue))
        self.assertEqual(sorted([key1_2, key1_4, key1_C, key1_F]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_mixed_depth(self):
        c_map = self.make_one_deep_two_prefix_map(chk_map._search_key_plain)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_aa = c_map._root_node._items[b'aa'].key()
        key1_ad = c_map._root_node._items[b'ad'].key()
        c_map2 = self.make_one_deep_one_prefix_map(chk_map._search_key_plain)
        c_map2._dump_tree()
        key2 = c_map2.key()
        key2_a = c_map2._root_node._items[b'a'].key()
        key2_b = c_map2._root_node._items[b'b'].key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([key2_b], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        diff = self.get_difference([key1], [key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key1], root_results)
        self.assertEqual([key2_a], diff._old_queue)
        self.assertEqual([key1_aa], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_yields_extra_deep_records(self):
        c_map = self.make_two_deep_map(chk_map._search_key_plain)
        c_map._dump_tree()
        key1 = c_map.key()
        key1_a = c_map._root_node._items[b'a'].key()
        c_map2 = self.get_map({(b'acc',): b'initial acc content', (b'ace',): b'initial ace content'}, maximum_size=100)
        self.assertEqualDiff("'' LeafNode\n      ('acc',) 'initial acc content'\n      ('ace',) 'initial ace content'\n", c_map2._dump_tree())
        key2 = c_map2.key()
        diff = self.get_difference([key2], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key2], root_results)
        self.assertEqual([key1_a], diff._old_queue)
        self.assertEqual({((b'acc',), b'initial acc content'), ((b'ace',), b'initial ace content')}, set(diff._new_item_queue))

    def test__read_all_roots_multiple_targets(self):
        c_map = self.make_root_only_map()
        key1 = c_map.key()
        c_map = self.make_one_deep_map()
        key2 = c_map.key()
        c_map._dump_tree()
        key2_c = c_map._root_node._items[b'c'].key()
        key2_d = c_map._root_node._items[b'd'].key()
        c_map.map((b'ccc',), b'new ccc value')
        key3 = c_map._save()
        key3_c = c_map._root_node._items[b'c'].key()
        diff = self.get_difference([key2, key3], [key1], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual(sorted([key2, key3]), sorted(root_results))
        self.assertEqual([], diff._old_queue)
        self.assertEqual(sorted([key2_c, key3_c, key2_d]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_no_old(self):
        c_map = self.make_two_deep_map()
        key1 = c_map.key()
        diff = self.get_difference([key1], [], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([key1], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        c_map2 = self.make_one_deep_map()
        key2 = c_map2.key()
        diff = self.get_difference([key1, key2], [], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual(sorted([key1, key2]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_no_old_16(self):
        c_map = self.make_two_deep_map(chk_map._search_key_16)
        key1 = c_map.key()
        diff = self.get_difference([key1], [], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual([key1], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        c_map2 = self.make_one_deep_map(chk_map._search_key_16)
        key2 = c_map2.key()
        diff = self.get_difference([key1, key2], [], chk_map._search_key_16)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([], root_results)
        self.assertEqual([], diff._old_queue)
        self.assertEqual(sorted([key1, key2]), sorted(diff._new_queue))
        self.assertEqual([], diff._new_item_queue)

    def test__read_all_roots_multiple_old(self):
        c_map = self.make_two_deep_map()
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items[b'a'].key()
        c_map.map((b'ccc',), b'new ccc value')
        key2 = c_map._save()
        key2_a = c_map._root_node._items[b'a'].key()
        c_map.map((b'add',), b'new add value')
        key3 = c_map._save()
        key3_a = c_map._root_node._items[b'a'].key()
        diff = self.get_difference([key3], [key1, key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key3], root_results)
        self.assertEqual([key1_a], diff._old_queue)
        self.assertEqual([key3_a], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)

    def test__process_next_old_batched_no_dupes(self):
        c_map = self.make_two_deep_map()
        key1 = c_map.key()
        c_map._dump_tree()
        key1_a = c_map._root_node._items[b'a'].key()
        key1_aa = c_map._root_node._items[b'a']._items[b'aa'].key()
        key1_ab = c_map._root_node._items[b'a']._items[b'ab'].key()
        key1_ac = c_map._root_node._items[b'a']._items[b'ac'].key()
        key1_ad = c_map._root_node._items[b'a']._items[b'ad'].key()
        c_map.map((b'aaa',), b'new aaa value')
        key2 = c_map._save()
        key2_a = c_map._root_node._items[b'a'].key()
        key2_aa = c_map._root_node._items[b'a']._items[b'aa'].key()
        c_map.map((b'acc',), b'new acc content')
        key3 = c_map._save()
        key3_a = c_map._root_node._items[b'a'].key()
        key3_ac = c_map._root_node._items[b'a']._items[b'ac'].key()
        diff = self.get_difference([key3], [key1, key2], chk_map._search_key_plain)
        root_results = [record.key for record in diff._read_all_roots()]
        self.assertEqual([key3], root_results)
        self.assertEqual(sorted([key1_a, key2_a]), sorted(diff._old_queue))
        self.assertEqual([key3_a], diff._new_queue)
        self.assertEqual([], diff._new_item_queue)
        diff._process_next_old()
        self.assertEqual(sorted([key1_aa, key1_ab, key1_ac, key1_ad, key2_aa]), sorted(diff._old_queue))