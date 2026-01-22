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
class TestPlainKnitContent(TestCase, KnitContentTestsMixin):

    def _make_content(self, lines):
        annotated_content = AnnotatedKnitContent(lines)
        return PlainKnitContent(annotated_content.text(), 'bogus')

    def test_annotate(self):
        content = self._make_content([])
        self.assertEqual(content.annotate(), [])
        content = self._make_content([('origin1', 'text1'), ('origin2', 'text2')])
        self.assertEqual(content.annotate(), [('bogus', 'text1'), ('bogus', 'text2')])

    def test_line_delta(self):
        content1 = self._make_content([('', 'a'), ('', 'b')])
        content2 = self._make_content([('', 'a'), ('', 'a'), ('', 'c')])
        self.assertEqual(content1.line_delta(content2), [(1, 2, 2, ['a', 'c'])])

    def test_line_delta_iter(self):
        content1 = self._make_content([('', 'a'), ('', 'b')])
        content2 = self._make_content([('', 'a'), ('', 'a'), ('', 'c')])
        it = content1.line_delta_iter(content2)
        self.assertEqual(next(it), (1, 2, 2, ['a', 'c']))
        self.assertRaises(StopIteration, next, it)