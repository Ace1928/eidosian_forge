from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
class ConversionTests(TestCaseWithTransport):

    def test_simple(self):
        wt = self.make_branch_and_tree('.')
        b = wt.branch
        with open('hello', 'w') as f:
            f.write('hello world')
        wt.add('hello')
        wt.commit(message='add hello', rev_id=b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bloe')
        wt.set_last_revision(b'bla')
        b.generate_revision_history(b'bla')
        with open('hello', 'w') as f:
            f.write('world')
        wt.commit(message='change hello', rev_id=b'bla2')
        wt.branch.repository.lock_write()
        newrev = CommitBuilderRevisionRewriter(wt.branch.repository)(b'bla2', b'bla4', (b'bloe',))
        self.assertEqual(b'bla4', newrev)
        self.assertTrue(wt.branch.repository.has_revision(newrev))
        self.assertEqual((b'bloe',), wt.branch.repository.get_parent_map([newrev])[newrev])
        self.assertEqual('bla2', wt.branch.repository.get_revision(newrev).properties['rebase-of'])
        wt.branch.repository.unlock()