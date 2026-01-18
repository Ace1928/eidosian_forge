from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_two_revisions_no_renames(self):
    wt = self.make_branch_and_tree('old')
    self.build_tree(['old/afile', 'old/notherfile'])
    wt.add(['afile'], ids=[b'somefileid'])
    wt.commit('bla', rev_id=b'oldparent')
    wt.add(['notherfile'])
    wt.commit('bla', rev_id=b'oldcommit')
    oldrepos = wt.branch.repository
    wt = self.make_branch_and_tree('new')
    self.build_tree(['new/afile', 'new/notherfile'])
    wt.add(['afile'], ids=[b'afileid'])
    wt.commit('bla', rev_id=b'newparent')
    wt.branch.repository.fetch(oldrepos)
    wt.branch.repository.lock_write()
    CommitBuilderRevisionRewriter(wt.branch.repository)(b'oldcommit', b'newcommit', (b'newparent',))
    wt.branch.repository.unlock()