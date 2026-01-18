import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
def test_reset_handler(self):
    from fastimport import commands
    [c1] = build_commit_graph(self.repo.object_store, [[1]])
    cmd = commands.ResetCommand(b'refs/heads/foo', c1.id)
    self.processor.reset_handler(cmd)
    self.assertEqual(c1.id, self.repo.get_refs()[b'refs/heads/foo'])
    self.assertEqual(c1.id, self.processor.last_commit)