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
class MockReadvFailingTransport(MockTransport):
    """Fail in the middle of a readv() result.

    This Transport will successfully yield the first two requested hunks, but
    raise NoSuchFile for the rest.
    """

    def readv(self, relpath, offsets):
        count = 0
        for result in MockTransport.readv(self, relpath, offsets):
            count += 1
            if count > 2:
                raise _mod_transport.NoSuchFile(relpath)
            yield result