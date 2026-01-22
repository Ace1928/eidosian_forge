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
def _add_content_nostoresha(self, add_lines):
    """When nostore_sha is supplied using old content raises."""
    vf = self.get_versionedfiles()
    empty_text = (b'a', [])
    sample_text_nl = (b'b', [b'foo\n', b'bar\n'])
    sample_text_no_nl = (b'c', [b'foo\n', b'bar'])
    shas = []
    for version, lines in (empty_text, sample_text_nl, sample_text_no_nl):
        if add_lines:
            sha, _, _ = vf.add_lines(self.get_simple_key(version), [], lines)
        else:
            sha, _, _ = vf.add_lines(self.get_simple_key(version), [], lines)
        shas.append(sha)
    for sha, (version, lines) in zip(shas, (empty_text, sample_text_nl, sample_text_no_nl)):
        new_key = self.get_simple_key(version + b'2')
        self.assertRaises(ExistingContent, vf.add_lines, new_key, [], lines, nostore_sha=sha)
        self.assertRaises(ExistingContent, vf.add_lines, new_key, [], lines, nostore_sha=sha)
        record = next(vf.get_record_stream([new_key], 'unordered', True))
        self.assertEqual('absent', record.storage_kind)