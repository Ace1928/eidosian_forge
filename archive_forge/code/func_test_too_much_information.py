import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_too_much_information(self):
    r = revision.Revision('1', properties={'bugs': 'http://example.com/bugs/1 fixed bar'})
    self.assertRaises(bugtracker.InvalidLineInBugsProperty, list, r.iter_bugs())