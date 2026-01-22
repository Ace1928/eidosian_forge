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
class MergeCasesMixin:

    def doMerge(self, base, a, b, mp):
        from textwrap import dedent

        def addcrlf(x):
            return x + b'\n'
        w = self.get_file()
        w.add_lines(b'text0', [], list(map(addcrlf, base)))
        w.add_lines(b'text1', [b'text0'], list(map(addcrlf, a)))
        w.add_lines(b'text2', [b'text0'], list(map(addcrlf, b)))
        self.log_contents(w)
        self.log('merge plan:')
        p = list(w.plan_merge(b'text1', b'text2'))
        for state, line in p:
            if line:
                self.log('%12s | %s' % (state, line[:-1]))
        self.log('merge:')
        mt = BytesIO()
        mt.writelines(w.weave_merge(p))
        mt.seek(0)
        self.log(mt.getvalue())
        mp = list(map(addcrlf, mp))
        self.assertEqual(mt.readlines(), mp)

    def testOneInsert(self):
        self.doMerge([], [b'aa'], [], [b'aa'])

    def testSeparateInserts(self):
        self.doMerge([b'aaa', b'bbb', b'ccc'], [b'aaa', b'xxx', b'bbb', b'ccc'], [b'aaa', b'bbb', b'yyy', b'ccc'], [b'aaa', b'xxx', b'bbb', b'yyy', b'ccc'])

    def testSameInsert(self):
        self.doMerge([b'aaa', b'bbb', b'ccc'], [b'aaa', b'xxx', b'bbb', b'ccc'], [b'aaa', b'xxx', b'bbb', b'yyy', b'ccc'], [b'aaa', b'xxx', b'bbb', b'yyy', b'ccc'])
    overlappedInsertExpected = [b'aaa', b'xxx', b'yyy', b'bbb']

    def testOverlappedInsert(self):
        self.doMerge([b'aaa', b'bbb'], [b'aaa', b'xxx', b'yyy', b'bbb'], [b'aaa', b'xxx', b'bbb'], self.overlappedInsertExpected)

    def testClashReplace(self):
        self.doMerge([b'aaa'], [b'xxx'], [b'yyy', b'zzz'], [b'<<<<<<< ', b'xxx', b'=======', b'yyy', b'zzz', b'>>>>>>> '])

    def testNonClashInsert1(self):
        self.doMerge([b'aaa'], [b'xxx', b'aaa'], [b'yyy', b'zzz'], [b'<<<<<<< ', b'xxx', b'aaa', b'=======', b'yyy', b'zzz', b'>>>>>>> '])

    def testNonClashInsert2(self):
        self.doMerge([b'aaa'], [b'aaa'], [b'yyy', b'zzz'], [b'yyy', b'zzz'])

    def testDeleteAndModify(self):
        """Clashing delete and modification.

        If one side modifies a region and the other deletes it then
        there should be a conflict with one side blank.
        """
        return
        self.doMerge([b'aaa', b'bbb', b'ccc'], [b'aaa', b'ddd', b'ccc'], [b'aaa', b'ccc'], [b'<<<<<<<< ', b'aaa', b'=======', b'>>>>>>> ', b'ccc'])

    def _test_merge_from_strings(self, base, a, b, expected):
        w = self.get_file()
        w.add_lines(b'text0', [], base.splitlines(True))
        w.add_lines(b'text1', [b'text0'], a.splitlines(True))
        w.add_lines(b'text2', [b'text0'], b.splitlines(True))
        self.log('merge plan:')
        p = list(w.plan_merge(b'text1', b'text2'))
        for state, line in p:
            if line:
                self.log('%12s | %s' % (state, line[:-1]))
        self.log('merge result:')
        result_text = b''.join(w.weave_merge(p))
        self.log(result_text)
        self.assertEqualDiff(result_text, expected)

    def test_weave_merge_conflicts(self):
        result = b''.join(self.get_file().weave_merge([('new-a', b'hello\n')]))
        self.assertEqual(result, b'hello\n')

    def test_deletion_extended(self):
        """One side deletes, the other deletes more.
        """
        base = b'            line 1\n            line 2\n            line 3\n            '
        a = b'            line 1\n            line 2\n            '
        b = b'            line 1\n            '
        result = b'            line 1\n<<<<<<< \n            line 2\n=======\n>>>>>>> \n            '
        self._test_merge_from_strings(base, a, b, result)

    def test_deletion_overlap(self):
        """Delete overlapping regions with no other conflict.

        Arguably it'd be better to treat these as agreement, rather than
        conflict, but for now conflict is safer.
        """
        base = b'            start context\n            int a() {}\n            int b() {}\n            int c() {}\n            end context\n            '
        a = b'            start context\n            int a() {}\n            end context\n            '
        b = b'            start context\n            int c() {}\n            end context\n            '
        result = b'            start context\n<<<<<<< \n            int a() {}\n=======\n            int c() {}\n>>>>>>> \n            end context\n            '
        self._test_merge_from_strings(base, a, b, result)

    def test_agreement_deletion(self):
        """Agree to delete some lines, without conflicts."""
        base = b'            start context\n            base line 1\n            base line 2\n            end context\n            '
        a = b'            start context\n            base line 1\n            end context\n            '
        b = b'            start context\n            base line 1\n            end context\n            '
        result = b'            start context\n            base line 1\n            end context\n            '
        self._test_merge_from_strings(base, a, b, result)

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