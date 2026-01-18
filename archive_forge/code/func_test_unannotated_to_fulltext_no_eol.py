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
def test_unannotated_to_fulltext_no_eol(self):
    """Test adapting unannotated knits to full texts.

        This is used for -> weaves, and for -> annotated knits.
        """
    f = self.get_knit(annotated=False)
    get_diamond_files(f, 1, trailing_eol=False)
    logged_vf = versionedfile.RecordingVersionedFilesDecorator(f)
    ft_data, delta_data = self.helpGetBytes(f, 'fulltext', _mod_knit.FTPlainToFullText(None), 'fulltext', _mod_knit.DeltaPlainToFullText(logged_vf))
    self.assertEqual(b'origin', ft_data)
    self.assertEqual(b'base\nleft\nright\nmerged', delta_data)
    self.assertEqual([('get_record_stream', [(b'left',)], 'unordered', True)], logged_vf.calls)