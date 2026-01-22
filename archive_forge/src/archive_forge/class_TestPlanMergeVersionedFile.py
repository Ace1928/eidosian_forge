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
class TestPlanMergeVersionedFile(TestCaseWithMemoryTransport):

    def setUp(self):
        super().setUp()
        mapper = PrefixMapper()
        factory = make_file_factory(True, mapper)
        self.vf1 = factory(self.get_transport('root-1'))
        self.vf2 = factory(self.get_transport('root-2'))
        self.plan_merge_vf = versionedfile._PlanMergeVersionedFile('root')
        self.plan_merge_vf.fallback_versionedfiles.extend([self.vf1, self.vf2])

    def test_add_lines(self):
        self.plan_merge_vf.add_lines((b'root', b'a:'), [], [])
        self.assertRaises(ValueError, self.plan_merge_vf.add_lines, (b'root', b'a'), [], [])
        self.assertRaises(ValueError, self.plan_merge_vf.add_lines, (b'root', b'a:'), None, [])
        self.assertRaises(ValueError, self.plan_merge_vf.add_lines, (b'root', b'a:'), [], None)

    def setup_abcde(self):
        self.vf1.add_lines((b'root', b'A'), [], [b'a'])
        self.vf1.add_lines((b'root', b'B'), [(b'root', b'A')], [b'b'])
        self.vf2.add_lines((b'root', b'C'), [], [b'c'])
        self.vf2.add_lines((b'root', b'D'), [(b'root', b'C')], [b'd'])
        self.plan_merge_vf.add_lines((b'root', b'E:'), [(b'root', b'B'), (b'root', b'D')], [b'e'])

    def test_get_parents(self):
        self.setup_abcde()
        self.assertEqual({(b'root', b'B'): ((b'root', b'A'),)}, self.plan_merge_vf.get_parent_map([(b'root', b'B')]))
        self.assertEqual({(b'root', b'D'): ((b'root', b'C'),)}, self.plan_merge_vf.get_parent_map([(b'root', b'D')]))
        self.assertEqual({(b'root', b'E:'): ((b'root', b'B'), (b'root', b'D'))}, self.plan_merge_vf.get_parent_map([(b'root', b'E:')]))
        self.assertEqual({}, self.plan_merge_vf.get_parent_map([(b'root', b'F')]))
        self.assertEqual({(b'root', b'B'): ((b'root', b'A'),), (b'root', b'D'): ((b'root', b'C'),), (b'root', b'E:'): ((b'root', b'B'), (b'root', b'D'))}, self.plan_merge_vf.get_parent_map([(b'root', b'B'), (b'root', b'D'), (b'root', b'E:'), (b'root', b'F')]))

    def test_get_record_stream(self):
        self.setup_abcde()

        def get_record(suffix):
            return next(self.plan_merge_vf.get_record_stream([(b'root', suffix)], 'unordered', True))
        self.assertEqual(b'a', get_record(b'A').get_bytes_as('fulltext'))
        self.assertEqual(b'a', b''.join(get_record(b'A').iter_bytes_as('chunked')))
        self.assertEqual(b'c', get_record(b'C').get_bytes_as('fulltext'))
        self.assertEqual(b'e', get_record(b'E:').get_bytes_as('fulltext'))
        self.assertEqual('absent', get_record('F').storage_kind)