from breezy.tests.per_repository import TestCaseWithRepository
def test_pack_empty_does_not_error(self):
    repo = self.make_repository('.')
    repo.pack()