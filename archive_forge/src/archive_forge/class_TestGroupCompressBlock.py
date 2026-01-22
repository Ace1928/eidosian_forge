import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class TestGroupCompressBlock(tests.TestCase):

    def make_block(self, key_to_text):
        """Create a GroupCompressBlock, filling it with the given texts."""
        compressor = groupcompress.GroupCompressor()
        start = 0
        for key in sorted(key_to_text):
            compressor.compress(key, [key_to_text[key]], len(key_to_text[key]), None)
        locs = {key: (start, end) for key, (start, _, end, _) in compressor.labels_deltas.items()}
        block = compressor.flush()
        raw_bytes = block.to_bytes()
        return (locs, groupcompress.GroupCompressBlock.from_bytes(raw_bytes))

    def test_from_empty_bytes(self):
        self.assertRaises(ValueError, groupcompress.GroupCompressBlock.from_bytes, b'')

    def test_from_minimal_bytes(self):
        block = groupcompress.GroupCompressBlock.from_bytes(b'gcb1z\n0\n0\n')
        self.assertIsInstance(block, groupcompress.GroupCompressBlock)
        self.assertIs(None, block._content)
        self.assertEqual(b'', block._z_content)
        block._ensure_content()
        self.assertEqual(b'', block._content)
        self.assertEqual(b'', block._z_content)
        block._ensure_content()

    def test_from_invalid(self):
        self.assertRaises(ValueError, groupcompress.GroupCompressBlock.from_bytes, b'this is not a valid header')

    def test_from_bytes(self):
        content = b'a tiny bit of content\n'
        z_content = zlib.compress(content)
        z_bytes = b'gcb1z\n%d\n%d\n%s' % (len(z_content), len(content), z_content)
        block = groupcompress.GroupCompressBlock.from_bytes(z_bytes)
        self.assertEqual(z_content, block._z_content)
        self.assertIs(None, block._content)
        self.assertEqual(len(z_content), block._z_content_length)
        self.assertEqual(len(content), block._content_length)
        block._ensure_content()
        self.assertEqual(z_content, block._z_content)
        self.assertEqual(content, block._content)

    def test_to_chunks(self):
        content_chunks = [b'this is some content\n', b'this content will be compressed\n']
        content_len = sum(map(len, content_chunks))
        content = b''.join(content_chunks)
        gcb = groupcompress.GroupCompressBlock()
        gcb.set_chunked_content(content_chunks, content_len)
        total_len, block_chunks = gcb.to_chunks()
        block_bytes = b''.join(block_chunks)
        self.assertEqual(gcb._z_content_length, len(gcb._z_content))
        self.assertEqual(total_len, len(block_bytes))
        self.assertEqual(gcb._content_length, content_len)
        expected_header = b'gcb1z\n%d\n%d\n' % (gcb._z_content_length, gcb._content_length)
        self.assertEqual(expected_header, block_chunks[0])
        self.assertStartsWith(block_bytes, expected_header)
        remaining_bytes = block_bytes[len(expected_header):]
        raw_bytes = zlib.decompress(remaining_bytes)
        self.assertEqual(content, raw_bytes)

    def test_to_bytes(self):
        content = b'this is some content\nthis content will be compressed\n'
        gcb = groupcompress.GroupCompressBlock()
        gcb.set_content(content)
        data = gcb.to_bytes()
        self.assertEqual(gcb._z_content_length, len(gcb._z_content))
        self.assertEqual(gcb._content_length, len(content))
        expected_header = b'gcb1z\n%d\n%d\n' % (gcb._z_content_length, gcb._content_length)
        self.assertStartsWith(data, expected_header)
        remaining_bytes = data[len(expected_header):]
        raw_bytes = zlib.decompress(remaining_bytes)
        self.assertEqual(content, raw_bytes)
        gcb = groupcompress.GroupCompressBlock()
        gcb.set_chunked_content([b'this is some content\nthis content will be compressed\n'], len(content))
        old_data = data
        data = gcb.to_bytes()
        self.assertEqual(old_data, data)

    def test_partial_decomp(self):
        content_chunks = []
        for i in range(2048):
            next_content = b'%d\nThis is a bit of duplicate text\n' % (i,)
            content_chunks.append(next_content)
            next_sha1 = osutils.sha_string(next_content)
            content_chunks.append(next_sha1 + b'\n')
        content = b''.join(content_chunks)
        self.assertEqual(158634, len(content))
        z_content = zlib.compress(content)
        self.assertEqual(57182, len(z_content))
        block = groupcompress.GroupCompressBlock()
        block._z_content_chunks = (z_content,)
        block._z_content_length = len(z_content)
        block._compressor_name = 'zlib'
        block._content_length = 158634
        self.assertIs(None, block._content)
        block._ensure_content(100)
        self.assertIsNot(None, block._content)
        self.assertTrue(len(block._content) >= 100)
        self.assertTrue(len(block._content) < 158634)
        self.assertEqualDiff(content[:len(block._content)], block._content)
        cur_len = len(block._content)
        block._ensure_content(cur_len - 10)
        self.assertEqual(cur_len, len(block._content))
        cur_len += 10
        block._ensure_content(cur_len)
        self.assertTrue(len(block._content) >= cur_len)
        self.assertTrue(len(block._content) < 158634)
        self.assertEqualDiff(content[:len(block._content)], block._content)
        block._ensure_content(158634)
        self.assertEqualDiff(content, block._content)
        self.assertIs(None, block._z_content_decompressor)

    def test__ensure_all_content(self):
        content_chunks = []
        for i in range(2048):
            next_content = b'%d\nThis is a bit of duplicate text\n' % (i,)
            content_chunks.append(next_content)
            next_sha1 = osutils.sha_string(next_content)
            content_chunks.append(next_sha1 + b'\n')
        content = b''.join(content_chunks)
        self.assertEqual(158634, len(content))
        z_content = zlib.compress(content)
        self.assertEqual(57182, len(z_content))
        block = groupcompress.GroupCompressBlock()
        block._z_content_chunks = (z_content,)
        block._z_content_length = len(z_content)
        block._compressor_name = 'zlib'
        block._content_length = 158634
        self.assertIs(None, block._content)
        block._ensure_content(158634)
        self.assertEqualDiff(content, block._content)
        self.assertIs(None, block._z_content_decompressor)

    def test__dump(self):
        dup_content = b'some duplicate content\nwhich is sufficiently long\n'
        key_to_text = {(b'1',): dup_content + b'1 unique\n', (b'2',): dup_content + b'2 extra special\n'}
        locs, block = self.make_block(key_to_text)
        self.assertEqual([(b'f', len(key_to_text[b'1',])), (b'd', 21, len(key_to_text[b'2',]), [(b'c', 2, len(dup_content)), (b'i', len(b'2 extra special\n'), b'')])], block._dump())