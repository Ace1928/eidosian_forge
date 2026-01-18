from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
def test_simple_unmarshall_rebase_plan(self):
    self.assertEqual(((1, b'bla'), {b'oldrev': (b'newrev', (b'newparent1', b'newparent2'))}), unmarshall_rebase_plan(b'# Bazaar rebase plan 1\n1 bla\noldrev newrev newparent1 newparent2\n'))