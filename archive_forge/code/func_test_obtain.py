from ..repo import MemoryRepo
from ..stash import Stash
from . import TestCase
def test_obtain(self):
    repo = MemoryRepo()
    stash = Stash.from_repo(repo)
    self.assertIsInstance(stash, Stash)