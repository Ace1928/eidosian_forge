import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
def test_two_nosha_delta(self):
    compressor = self.compressor()
    text = b'strange\ncommon long line\nthat needs a 16 byte match\n'
    sha1_1, _, _, _ = compressor.compress(('label',), [text], len(text), None)
    expected_lines = list(compressor.chunks)
    text = b'common long line\nthat needs a 16 byte match\ndifferent\n'
    sha1_2, start_point, end_point, _ = compressor.compress(('newlabel',), [text], len(text), None)
    self.assertEqual(sha_string(text), sha1_2)
    expected_lines.extend([b'd\x0f', b'6', b'\x91\n,', b'\ndifferent\n'])
    self.assertEqualDiffEncoded(expected_lines, compressor.chunks)
    self.assertEqual(sum(map(len, expected_lines)), end_point)