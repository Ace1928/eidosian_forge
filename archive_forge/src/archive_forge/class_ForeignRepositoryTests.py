from breezy.tests import TestCase, TestCaseWithTransport
class ForeignRepositoryTests(TestCaseWithTransport):
    """Basic tests for foreign repository implementations.

    These tests mainly make sure that the implementation covers the required
    bits of the API and returns semi-reasonable values, that are
    at least of the expected types and in the expected ranges.
    """
    repository_factory = None

    def make_repository(self):
        return self.repository_factory.make_repository(self.get_transport())

    def test_make_working_trees(self):
        """Test that Repository.make_working_trees() returns a boolean."""
        repo = self.make_repository()
        self.assertIsInstance(repo.make_working_trees(), bool)

    def test_get_physical_lock_status(self):
        """Test that a new repository is not locked by default."""
        repo = self.make_repository()
        self.assertFalse(repo.get_physical_lock_status())

    def test_is_shared(self):
        """Test that is_shared() returns a bool."""
        repo = self.make_repository()
        self.assertIsInstance(repo.is_shared(), bool)

    def test_gather_stats(self):
        """Test that gather_stats() will at least return a dictionary
        with the required keys."""
        repo = self.make_repository()
        stats = repo.gather_stats()
        self.assertIsInstance(stats, dict)