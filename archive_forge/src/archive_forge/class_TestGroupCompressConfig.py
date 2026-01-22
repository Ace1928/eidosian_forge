import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class TestGroupCompressConfig(tests.TestCaseWithTransport):

    def make_test_vf(self):
        t = self.get_transport('.')
        t.ensure_base()
        factory = groupcompress.make_pack_factory(graph=True, delta=False, keylength=1, inconsistency_fatal=True)
        vf = factory(t)
        self.addCleanup(groupcompress.cleanup_pack_group, vf)
        return vf

    def test_max_bytes_to_index_default(self):
        vf = self.make_test_vf()
        gc = vf._make_group_compressor()
        self.assertEqual(vf._DEFAULT_MAX_BYTES_TO_INDEX, vf._max_bytes_to_index)
        if isinstance(gc, groupcompress.PyrexGroupCompressor):
            self.assertEqual(vf._DEFAULT_MAX_BYTES_TO_INDEX, gc._delta_index._max_bytes_to_index)

    def test_max_bytes_to_index_in_config(self):
        c = config.GlobalConfig()
        c.set_user_option('bzr.groupcompress.max_bytes_to_index', '10000')
        vf = self.make_test_vf()
        gc = vf._make_group_compressor()
        self.assertEqual(10000, vf._max_bytes_to_index)
        if isinstance(gc, groupcompress.PyrexGroupCompressor):
            self.assertEqual(10000, gc._delta_index._max_bytes_to_index)

    def test_max_bytes_to_index_bad_config(self):
        c = config.GlobalConfig()
        c.set_user_option('bzr.groupcompress.max_bytes_to_index', 'boogah')
        vf = self.make_test_vf()
        gc = vf._make_group_compressor()
        self.assertEqual(vf._DEFAULT_MAX_BYTES_TO_INDEX, vf._max_bytes_to_index)
        if isinstance(gc, groupcompress.PyrexGroupCompressor):
            self.assertEqual(vf._DEFAULT_MAX_BYTES_TO_INDEX, gc._delta_index._max_bytes_to_index)