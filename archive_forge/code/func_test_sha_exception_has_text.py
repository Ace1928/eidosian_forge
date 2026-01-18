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
def test_sha_exception_has_text(self):
    source = self.make_test_knit()
    target = self.make_test_knit(name='target')
    if not source._max_delta_chain:
        raise TestNotApplicable('cannot get delta-caused sha failures without deltas.')
    basis = (b'basis',)
    broken = (b'broken',)
    source.add_lines(basis, (), [b'foo\n'])
    source.add_lines(broken, (basis,), [b'foo\n', b'bar\n'])
    target.add_lines(basis, (), [b'gam\n'])
    target.insert_record_stream(source.get_record_stream([broken], 'unordered', False))
    err = self.assertRaises(KnitCorrupt, next(target.get_record_stream([broken], 'unordered', True)).get_bytes_as, 'chunked')
    self.assertEqual([b'gam\n', b'bar\n'], err.content)
    self.assertStartsWith(str(err), 'Knit ')