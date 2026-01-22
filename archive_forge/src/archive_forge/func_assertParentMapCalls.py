from breezy import tests, urlutils
from breezy.bzr import remote
from breezy.tests.per_repository import TestCaseWithRepository
def assertParentMapCalls(self, expected):
    """Check that self.hpss_calls has the expected get_parent_map calls."""
    get_parent_map_calls = []
    for c in self.hpss_calls:
        self.assertEqual(b'Repository.get_parent_map', c.call.method)
        args = c.call.args
        location = args[0]
        self.assertEqual(b'include-missing:', args[1])
        revisions = sorted(args[2:])
        get_parent_map_calls.append((location, revisions))
    self.assertEqual(expected, get_parent_map_calls)