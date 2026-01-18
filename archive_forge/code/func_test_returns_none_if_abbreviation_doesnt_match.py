from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_returns_none_if_abbreviation_doesnt_match(self):
    """The get() method should return None if the given abbreviated name
        doesn't match the tracker's abbreviation.
        """
    tracker = bugtracker.ProjectIntegerBugTracker('xxx', 'http://bugs.example.com/{project}/{id}')
    branch = self.make_branch('some_branch')
    self.assertIs(None, tracker.get('yyy', branch))