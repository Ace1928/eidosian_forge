import gzip
import sys
from io import BytesIO
from patiencediff import PatienceSequenceMatcher
from ... import errors, multiparent, osutils, tests
from ... import transport as _mod_transport
from ...tests import (TestCase, TestCaseWithMemoryTransport,
from .. import knit, knitpack_repo, pack, pack_repo
from ..index import *
from ..knit import (AnnotatedKnitContent, KnitContent, KnitCorrupt,
from ..versionedfile import (AbsentContentFactory, ConstantMapper,
def test_record_delta_removes_basis(self):
    ann = self.make_annotator()
    ann._expand_record((b'parent-id',), (), None, [b'line1\n', b'line2\n'], ('fulltext', False))
    ann._num_compression_children[b'parent-id'] = 2