from ... import errors, osutils, tests
from .. import chk_map, groupcompress
from ..chk_map import CHKMap, InternalNode, LeafNode, Node
from ..static_tuple import StaticTuple
class TestTestCaseWithExampleMaps(TestCaseWithExampleMaps):
    """Actual tests for the provided examples."""

    def test_root_only_map_plain(self):
        c_map = self.make_root_only_map()
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('abb',) 'initial abb content'\n", c_map._dump_tree())

    def test_root_only_map_16(self):
        c_map = self.make_root_only_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('abb',) 'initial abb content'\n", c_map._dump_tree())

    def test_one_deep_map_plain(self):
        c_map = self.make_one_deep_map()
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('abb',) 'initial abb content'\n  'c' LeafNode\n      ('ccc',) 'initial ccc content'\n  'd' LeafNode\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_one_deep_map_16(self):
        c_map = self.make_one_deep_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  '2' LeafNode\n      ('ccc',) 'initial ccc content'\n  '4' LeafNode\n      ('abb',) 'initial abb content'\n  'F' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_root_only_aaa_ddd_plain(self):
        c_map = self.make_root_only_aaa_ddd_map()
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_root_only_aaa_ddd_16(self):
        c_map = self.make_root_only_aaa_ddd_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' LeafNode\n      ('aaa',) 'initial aaa content'\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_two_deep_map_plain(self):
        c_map = self.make_two_deep_map()
        self.assertEqualDiff("'' InternalNode\n  'a' InternalNode\n    'aa' LeafNode\n      ('aaa',) 'initial aaa content'\n    'ab' LeafNode\n      ('abb',) 'initial abb content'\n    'ac' LeafNode\n      ('acc',) 'initial acc content'\n      ('ace',) 'initial ace content'\n    'ad' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n  'c' LeafNode\n      ('ccc',) 'initial ccc content'\n  'd' LeafNode\n      ('ddd',) 'initial ddd content'\n", c_map._dump_tree())

    def test_two_deep_map_16(self):
        c_map = self.make_two_deep_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  '2' LeafNode\n      ('acc',) 'initial acc content'\n      ('ccc',) 'initial ccc content'\n  '4' LeafNode\n      ('abb',) 'initial abb content'\n  'C' LeafNode\n      ('ace',) 'initial ace content'\n  'F' InternalNode\n    'F0' LeafNode\n      ('aaa',) 'initial aaa content'\n    'F3' LeafNode\n      ('adl',) 'initial adl content'\n    'F4' LeafNode\n      ('adh',) 'initial adh content'\n    'FB' LeafNode\n      ('ddd',) 'initial ddd content'\n    'FD' LeafNode\n      ('add',) 'initial add content'\n", c_map._dump_tree())

    def test_one_deep_two_prefix_map_plain(self):
        c_map = self.make_one_deep_two_prefix_map()
        self.assertEqualDiff("'' InternalNode\n  'aa' LeafNode\n      ('aaa',) 'initial aaa content'\n  'ad' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n", c_map._dump_tree())

    def test_one_deep_two_prefix_map_16(self):
        c_map = self.make_one_deep_two_prefix_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  'F0' LeafNode\n      ('aaa',) 'initial aaa content'\n  'F3' LeafNode\n      ('adl',) 'initial adl content'\n  'F4' LeafNode\n      ('adh',) 'initial adh content'\n  'FD' LeafNode\n      ('add',) 'initial add content'\n", c_map._dump_tree())

    def test_one_deep_one_prefix_map_plain(self):
        c_map = self.make_one_deep_one_prefix_map()
        self.assertEqualDiff("'' InternalNode\n  'a' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n  'b' LeafNode\n      ('bbb',) 'initial bbb content'\n", c_map._dump_tree())

    def test_one_deep_one_prefix_map_16(self):
        c_map = self.make_one_deep_one_prefix_map(search_key_func=chk_map._search_key_16)
        self.assertEqualDiff("'' InternalNode\n  '4' LeafNode\n      ('bbb',) 'initial bbb content'\n  'F' LeafNode\n      ('add',) 'initial add content'\n      ('adh',) 'initial adh content'\n      ('adl',) 'initial adl content'\n", c_map._dump_tree())