import itertools
from gzip import GzipFile
from io import BytesIO
from ... import errors
from ... import graph as _mod_graph
from ... import osutils, progress, transport, ui
from ...errors import RevisionAlreadyPresent, RevisionNotPresent
from ...tests import (TestCase, TestCaseWithMemoryTransport, TestNotApplicable,
from ...tests.http_utils import TestCaseWithWebserver
from ...tests.scenarios import load_tests_apply_scenarios
from ...transport.memory import MemoryTransport
from .. import groupcompress
from .. import knit as _mod_knit
from .. import versionedfile as versionedfile
from ..knit import cleanup_pack_knit, make_file_factory, make_pack_factory
from ..versionedfile import (ChunkedContentFactory, ConstantMapper,
from ..weave import WeaveFile, WeaveInvalidChecksum
from ..weavefile import write_weave
class TestOrderingVersionedFilesDecorator(TestCaseWithMemoryTransport):

    def get_ordering_vf(self, key_priority):
        builder = self.make_branch_builder('test')
        builder.start_series()
        builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', None))], revision_id=b'A')
        builder.build_snapshot([b'A'], [], revision_id=b'B')
        builder.build_snapshot([b'B'], [], revision_id=b'C')
        builder.build_snapshot([b'C'], [], revision_id=b'D')
        builder.finish_series()
        b = builder.get_branch()
        b.lock_read()
        self.addCleanup(b.unlock)
        vf = b.repository.inventories
        return versionedfile.OrderingVersionedFilesDecorator(vf, key_priority)

    def test_get_empty(self):
        vf = self.get_ordering_vf({})
        self.assertEqual([], vf.calls)

    def test_get_record_stream_topological(self):
        vf = self.get_ordering_vf({(b'A',): 3, (b'B',): 2, (b'C',): 4, (b'D',): 1})
        request_keys = [(b'B',), (b'C',), (b'D',), (b'A',)]
        keys = [r.key for r in vf.get_record_stream(request_keys, 'topological', False)]
        self.assertEqual([(b'A',), (b'B',), (b'C',), (b'D',)], keys)
        self.assertEqual([('get_record_stream', request_keys, 'topological', False)], vf.calls)

    def test_get_record_stream_ordered(self):
        vf = self.get_ordering_vf({(b'A',): 3, (b'B',): 2, (b'C',): 4, (b'D',): 1})
        request_keys = [(b'B',), (b'C',), (b'D',), (b'A',)]
        keys = [r.key for r in vf.get_record_stream(request_keys, 'unordered', False)]
        self.assertEqual([(b'D',), (b'B',), (b'A',), (b'C',)], keys)
        self.assertEqual([('get_record_stream', request_keys, 'unordered', False)], vf.calls)

    def test_get_record_stream_implicit_order(self):
        vf = self.get_ordering_vf({(b'B',): 2, (b'D',): 1})
        request_keys = [(b'B',), (b'C',), (b'D',), (b'A',)]
        keys = [r.key for r in vf.get_record_stream(request_keys, 'unordered', False)]
        self.assertEqual([(b'A',), (b'C',), (b'D',), (b'B',)], keys)
        self.assertEqual([('get_record_stream', request_keys, 'unordered', False)], vf.calls)