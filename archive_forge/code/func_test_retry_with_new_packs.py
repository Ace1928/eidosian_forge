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
def test_retry_with_new_packs(self):
    fake_exc_info = ('{exc type}', '{exc value}', '{exc traceback}')
    error = pack_repo.RetryWithNewPacks('{context}', reload_occurred=False, exc_info=fake_exc_info)
    self.assertEqual('Pack files have changed, reload and retry. context: {context} {exc value}', str(error))