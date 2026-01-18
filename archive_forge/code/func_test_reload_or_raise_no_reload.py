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
def test_reload_or_raise_no_reload(self):
    access = pack_repo._DirectPackAccess({}, reload_func=None)
    retry_exc = self.make_retry_exception()
    self.assertRaises(_TestException, access.reload_or_raise, retry_exc)