from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
class TestMergeFileHistory(TestCaseWithTransport):

    def setUp(self):
        super().setUp()
        wt1 = self.make_branch_and_tree('br1')
        br1 = wt1.branch
        self.build_tree_contents([('br1/file', b'original contents\n')])
        wt1.add('file', ids=b'this-file-id')
        wt1.commit(message='rev 1-1', rev_id=b'1-1')
        dir_2 = br1.controldir.sprout('br2')
        br2 = dir_2.open_branch()
        wt2 = dir_2.open_workingtree()
        self.build_tree_contents([('br1/file', b'original from 1\n')])
        wt1.commit(message='rev 1-2', rev_id=b'1-2')
        self.build_tree_contents([('br1/file', b'agreement\n')])
        wt1.commit(message='rev 1-3', rev_id=b'1-3')
        self.build_tree_contents([('br2/file', b'contents in 2\n')])
        wt2.commit(message='rev 2-1', rev_id=b'2-1')
        self.build_tree_contents([('br2/file', b'agreement\n')])
        wt2.commit(message='rev 2-2', rev_id=b'2-2')

    def test_merge_fetches_file_history(self):
        """Merge brings across file histories"""
        br2 = Branch.open('br2')
        br1 = Branch.open('br1')
        wt2 = WorkingTree.open('br2').merge_from_branch(br1)
        br2.lock_read()
        self.addCleanup(br2.unlock)
        for rev_id, text in [(b'1-2', b'original from 1\n'), (b'1-3', b'agreement\n'), (b'2-1', b'contents in 2\n'), (b'2-2', b'agreement\n')]:
            self.assertEqualDiff(br2.repository.revision_tree(rev_id).get_file_text('file'), text)