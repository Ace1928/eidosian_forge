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
def test_read_utf8_parents(self):
    unicode_revision_id = 'version-А'
    utf8_revision_id = unicode_revision_id.encode('utf-8')
    transport = MockTransport([_KndxIndex.HEADER, b'version option 0 1 .%s :' % (utf8_revision_id,)])
    index = self.get_knit_index(transport, 'filename', 'r')
    self.assertEqual({(b'version',): ((utf8_revision_id,),)}, index.get_parent_map(index.keys()))