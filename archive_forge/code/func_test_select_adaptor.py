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
def test_select_adaptor(self):
    """Test expected adapters exist."""
    scenarios = [('knit-delta-gz', 'fulltext', _mod_knit.DeltaPlainToFullText), ('knit-delta-gz', 'lines', _mod_knit.DeltaPlainToFullText), ('knit-delta-gz', 'chunked', _mod_knit.DeltaPlainToFullText), ('knit-ft-gz', 'fulltext', _mod_knit.FTPlainToFullText), ('knit-ft-gz', 'lines', _mod_knit.FTPlainToFullText), ('knit-ft-gz', 'chunked', _mod_knit.FTPlainToFullText), ('knit-annotated-delta-gz', 'knit-delta-gz', _mod_knit.DeltaAnnotatedToUnannotated), ('knit-annotated-delta-gz', 'fulltext', _mod_knit.DeltaAnnotatedToFullText), ('knit-annotated-ft-gz', 'knit-ft-gz', _mod_knit.FTAnnotatedToUnannotated), ('knit-annotated-ft-gz', 'fulltext', _mod_knit.FTAnnotatedToFullText), ('knit-annotated-ft-gz', 'lines', _mod_knit.FTAnnotatedToFullText), ('knit-annotated-ft-gz', 'chunked', _mod_knit.FTAnnotatedToFullText)]
    for source, requested, klass in scenarios:
        adapter_factory = versionedfile.adapter_registry.get((source, requested))
        adapter = adapter_factory(None)
        self.assertIsInstance(adapter, klass)