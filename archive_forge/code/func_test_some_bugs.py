import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_some_bugs(self):
    r = revision.Revision('1', properties={'bugs': bugtracker.encode_fixes_bug_urls([('http://example.com/bugs/1', 'fixed'), ('http://launchpad.net/bugs/1234', 'fixed')])})
    self.assertEqual([('http://example.com/bugs/1', bugtracker.FIXED), ('http://launchpad.net/bugs/1234', bugtracker.FIXED)], list(r.iter_bugs()))