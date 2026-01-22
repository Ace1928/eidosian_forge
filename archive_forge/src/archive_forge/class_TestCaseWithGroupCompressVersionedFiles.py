import zlib
from ... import config, errors, osutils, tests, trace
from ...osutils import sha_string
from ...tests.scenarios import load_tests_apply_scenarios
from .. import btree_index, groupcompress
from .. import index as _mod_index
from .. import knit, versionedfile
from .test__groupcompress import compiled_groupcompress_feature
class TestCaseWithGroupCompressVersionedFiles(tests.TestCaseWithMemoryTransport):

    def make_test_vf(self, create_graph, keylength=1, do_cleanup=True, dir='.', inconsistency_fatal=True):
        t = self.get_transport(dir)
        t.ensure_base()
        vf = groupcompress.make_pack_factory(graph=create_graph, delta=False, keylength=keylength, inconsistency_fatal=inconsistency_fatal)(t)
        if do_cleanup:
            self.addCleanup(groupcompress.cleanup_pack_group, vf)
        return vf