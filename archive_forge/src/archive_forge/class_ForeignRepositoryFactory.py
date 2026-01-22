from breezy.tests import TestCase, TestCaseWithTransport
class ForeignRepositoryFactory:
    """Factory of repository for ForeignRepositoryTests."""

    def make_repository(self, transport):
        """Create a new, valid, repository. May or may not contain
        data."""
        raise NotImplementedError(self.make_repository)