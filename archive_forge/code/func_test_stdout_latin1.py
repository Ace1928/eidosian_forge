import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
def test_stdout_latin1(self):
    self.overrideAttr(osutils, '_cached_user_encoding', 'latin-1')
    working_tree = self.make_uncommitted_tree()
    stdout, stderr = self.run_bzr('status')
    expected = 'added:\n  hellØ\n'
    self.assertEqual(stdout, expected)