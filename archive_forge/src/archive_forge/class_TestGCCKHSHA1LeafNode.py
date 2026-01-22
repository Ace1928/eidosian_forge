import binascii
import bisect
from ... import tests
from .test_btree_index import compiled_btreeparser_feature
class TestGCCKHSHA1LeafNode(TestBtreeSerializer):

    def assertInvalid(self, data):
        """Ensure that we get a proper error when trying to parse invalid bytes.

        (mostly this is testing that bad input doesn't cause us to segfault)
        """
        self.assertRaises((ValueError, TypeError), self.module._parse_into_chk, data, 1, 0)

    def test_non_bytes(self):
        self.assertInvalid('type=leaf\n')

    def test_not_leaf(self):
        self.assertInvalid(b'type=internal\n')

    def test_empty_leaf(self):
        leaf = self.module._parse_into_chk(b'type=leaf\n', 1, 0)
        self.assertEqual(0, len(leaf))
        self.assertEqual([], leaf.all_items())
        self.assertEqual([], leaf.all_keys())
        self.assertFalse(('key',) in leaf)

    def test_one_key_leaf(self):
        leaf = self.module._parse_into_chk(_one_key_content, 1, 0)
        self.assertEqual(1, len(leaf))
        sha_key = (b'sha1:' + _hex_form,)
        self.assertEqual([sha_key], leaf.all_keys())
        self.assertEqual([(sha_key, (b'1 2 3 4', ()))], leaf.all_items())
        self.assertTrue(sha_key in leaf)

    def test_large_offsets(self):
        leaf = self.module._parse_into_chk(_large_offsets, 1, 0)
        self.assertEqual([b'12345678901 1234567890 0 1', b'2147483648 2147483647 0 1', b'4294967296 4294967295 4294967294 1'], [x[1][0] for x in leaf.all_items()])

    def test_many_key_leaf(self):
        leaf = self.module._parse_into_chk(_multi_key_content, 1, 0)
        self.assertEqual(8, len(leaf))
        all_keys = leaf.all_keys()
        self.assertEqual(8, len(leaf.all_keys()))
        for idx, key in enumerate(all_keys):
            self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])

    def test_common_shift(self):
        leaf = self.module._parse_into_chk(_multi_key_content, 1, 0)
        self.assertEqual(19, leaf.common_shift)
        lst = [1, 13, 28, 180, 190, 193, 210, 239]
        offsets = leaf._get_offsets()
        self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
        for idx, val in enumerate(lst):
            self.assertEqual(idx, offsets[val])
        for idx, key in enumerate(leaf.all_keys()):
            self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])

    def test_multi_key_same_offset(self):
        leaf = self.module._parse_into_chk(_multi_key_same_offset, 1, 0)
        self.assertEqual(24, leaf.common_shift)
        offsets = leaf._get_offsets()
        lst = [8, 200, 205, 205, 205, 205, 206, 206]
        self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
        for val in lst:
            self.assertEqual(lst.index(val), offsets[val])
        for idx, key in enumerate(leaf.all_keys()):
            self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])

    def test_all_common_prefix(self):
        leaf = self.module._parse_into_chk(_common_32_bits, 1, 0)
        self.assertEqual(0, leaf.common_shift)
        lst = [120] * 8
        offsets = leaf._get_offsets()
        self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
        for val in lst:
            self.assertEqual(lst.index(val), offsets[val])
        for idx, key in enumerate(leaf.all_keys()):
            self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])

    def test_many_entries(self):
        lines = [b'type=leaf\n']
        for i in range(500):
            key_str = b'sha1:%04x%s' % (i, _hex_form[:36])
            key = (key_str,)
            lines.append(b'%s\x00\x00%d %d %d %d\n' % (key_str, i, i, i, i))
        data = b''.join(lines)
        leaf = self.module._parse_into_chk(data, 1, 0)
        self.assertEqual(24 - 7, leaf.common_shift)
        offsets = leaf._get_offsets()
        lst = [x // 2 for x in range(500)]
        expected_offsets = [x * 2 for x in range(128)] + [255] * 129
        self.assertEqual(expected_offsets, offsets)
        lst = lst[:255]
        self.assertEqual([bisect.bisect_left(lst, x) for x in range(0, 257)], offsets)
        for val in lst:
            self.assertEqual(lst.index(val), offsets[val])
        for idx, key in enumerate(leaf.all_keys()):
            self.assertEqual(b'%d' % idx, leaf[key][0].split()[0])

    def test__sizeof__(self):
        leaf0 = self.module._parse_into_chk(b'type=leaf\n', 1, 0)
        leaf1 = self.module._parse_into_chk(_one_key_content, 1, 0)
        leafN = self.module._parse_into_chk(_multi_key_content, 1, 0)
        sizeof_1 = leaf1.__sizeof__() - leaf0.__sizeof__()
        self.assertTrue(sizeof_1 > 0)
        sizeof_N = leafN.__sizeof__() - leaf0.__sizeof__()
        self.assertEqual(sizeof_1 * len(leafN), sizeof_N)