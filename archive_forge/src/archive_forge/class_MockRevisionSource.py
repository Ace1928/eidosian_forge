import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
class MockRevisionSource:
    """A RevisionSource that takes a pregenerated graph.

    This is useful for testing revision graph algorithms where
    the actual branch existing is irrelevant.
    """

    def __init__(self, full_graph):
        self._full_graph = full_graph

    def get_revision_graph_with_ghosts(self, revision_ids):
        return self._full_graph