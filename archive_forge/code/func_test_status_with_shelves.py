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
def test_status_with_shelves(self):
    """Ensure that _show_shelve_summary handler works.
        """
    wt = self.make_branch_and_tree('.')
    self.build_tree(['hello.c'])
    wt.add('hello.c')
    self.run_bzr(['shelve', '--all', '-m', 'foo'])
    self.build_tree(['bye.c'])
    wt.add('bye.c')
    self.assertStatus(['added:\n', '  bye.c\n', '1 shelf exists. See "brz shelve --list" for details.\n'], wt)
    self.run_bzr(['shelve', '--all', '-m', 'bar'])
    self.build_tree(['eggs.c', 'spam.c'])
    wt.add('eggs.c')
    wt.add('spam.c')
    self.assertStatus(['added:\n', '  eggs.c\n', '  spam.c\n', '2 shelves exist. See "brz shelve --list" for details.\n'], wt)
    self.assertStatus(['added:\n', '  spam.c\n'], wt, specific_files=['spam.c'])