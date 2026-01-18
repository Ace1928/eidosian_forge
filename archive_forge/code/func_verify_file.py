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
def verify_file(f):
    versions = f.versions()
    self.assertTrue(b'r0' in versions)
    self.assertTrue(b'r1' in versions)
    self.assertTrue(b'r2' in versions)
    self.assertEqual(f.get_lines(b'r0'), [b'a\n', b'b\n'])
    self.assertEqual(f.get_lines(b'r1'), [b'b\n', b'c\n'])
    self.assertEqual(f.get_lines(b'r2'), [b'c\n', b'd\n'])
    self.assertEqual(3, f.num_versions())
    origins = f.annotate(b'r1')
    self.assertEqual(origins[0][0], b'r0')
    self.assertEqual(origins[1][0], b'r1')
    origins = f.annotate(b'r2')
    self.assertEqual(origins[0][0], b'r1')
    self.assertEqual(origins[1][0], b'r2')