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
def assertStreamMetaEqual(self, records, expected, stream):
    """Assert that streams expected and stream have the same records.

        :param records: A list to collect the seen records.
        :return: A generator of the records in stream.
        """
    for record, ref_record in zip(stream, expected):
        records.append(record)
        self.assertEqual(ref_record.key, record.key)
        self.assertEqual(ref_record.storage_kind, record.storage_kind)
        self.assertEqual(ref_record.parents, record.parents)
        yield record