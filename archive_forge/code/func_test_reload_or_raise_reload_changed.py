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
def test_reload_or_raise_reload_changed(self):
    reload_called, reload_func = self.make_reload_func(return_val=True)
    access = pack_repo._DirectPackAccess({}, reload_func=reload_func)
    retry_exc = self.make_retry_exception()
    access.reload_or_raise(retry_exc)
    self.assertEqual([1], reload_called)
    retry_exc.reload_occurred = True
    access.reload_or_raise(retry_exc)
    self.assertEqual([2], reload_called)