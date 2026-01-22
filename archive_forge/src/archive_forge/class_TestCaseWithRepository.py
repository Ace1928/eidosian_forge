from breezy import repository
from breezy.bzr.remote import RemoteRepositoryFormat
from breezy.tests import default_transport, multiply_tests, test_server
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import memory
class TestCaseWithRepository(TestCaseWithControlDir):

    def get_default_format(self):
        format = self.repository_format._matchingcontroldir
        self.assertEqual(format.repository_format, self.repository_format)
        return format

    def make_repository(self, relpath, shared=None, format=None):
        format = self.resolve_format(format)
        repo = super().make_repository(relpath, shared=shared, format=format)
        if format is None or format.repository_format is self.repository_format:
            if getattr(self, 'repository_to_test_repository', None):
                repo = self.repository_to_test_repository(repo)
        return repo