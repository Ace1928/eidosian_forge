import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
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