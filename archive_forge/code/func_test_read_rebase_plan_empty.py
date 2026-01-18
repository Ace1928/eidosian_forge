from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_read_rebase_plan_empty(self):
    self.wt._transport.put_bytes(REBASE_PLAN_FILENAME, b'')
    self.assertRaises(NoSuchFile, self.state.read_plan)