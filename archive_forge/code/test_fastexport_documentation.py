import stat
from io import BytesIO
from dulwich.tests import SkipTest, TestCase
from ..object_store import MemoryObjectStore
from ..objects import ZERO_SHA, Blob, Commit, Tree
from ..repo import MemoryRepo
from .utils import build_commit_graph
Create a trivial commit with the specified file commands.

        Args:
          file_cmds: File commands to run.
        Returns: The created commit object
        