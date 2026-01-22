from breezy import repository
from breezy.bzr import remote
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack5
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
class TestCaseWithRepositoryCHK(TestCaseWithRepository):

    def make_repository(self, path, format=None):
        TestCaseWithRepository.make_repository(self, path, format=format)
        return repository.Repository.open(self.get_transport(path).base)