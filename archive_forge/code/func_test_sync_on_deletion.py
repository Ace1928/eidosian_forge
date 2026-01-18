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
def test_sync_on_deletion(self):
    """Specific case of merge where we can synchronize incorrectly.

        A previous version of the weave merge concluded that the two versions
        agreed on deleting line 2, and this could be a synchronization point.
        Line 1 was then considered in isolation, and thought to be deleted on
        both sides.

        It's better to consider the whole thing as a disagreement region.
        """
    base = b'            start context\n            base line 1\n            base line 2\n            end context\n            '
    a = b"            start context\n            base line 1\n            a's replacement line 2\n            end context\n            "
    b = b'            start context\n            b replaces\n            both lines\n            end context\n            '
    result = b"            start context\n<<<<<<< \n            base line 1\n            a's replacement line 2\n=======\n            b replaces\n            both lines\n>>>>>>> \n            end context\n            "
    self._test_merge_from_strings(base, a, b, result)