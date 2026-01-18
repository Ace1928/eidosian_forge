from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_read_rebase_plan(self):
    self.wt._transport.put_bytes(REBASE_PLAN_FILENAME, b'# Bazaar rebase plan 1\n1 bla\noldrev newrev newparent1 newparent2\n')
    self.assertEqual(((1, b'bla'), {b'oldrev': (b'newrev', (b'newparent1', b'newparent2'))}), self.state.read_plan())