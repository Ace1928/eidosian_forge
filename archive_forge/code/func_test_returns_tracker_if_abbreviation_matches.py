from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_returns_tracker_if_abbreviation_matches(self):
    """The get() method should return an instance of the tracker if the
        given abbreviation matches the tracker's abbreviated name.
        """
    tracker = bugtracker.ProjectIntegerBugTracker('xxx', 'http://bugs.example.com/{project}/{id}')
    branch = self.make_branch('some_branch')
    self.assertIs(tracker, tracker.get('xxx', branch))