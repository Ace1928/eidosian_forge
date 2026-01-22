import datetime
import os
import shutil
import tempfile
import time
import types
import warnings
from dulwich.tests import SkipTest
from ..index import commit_tree
from ..objects import Commit, FixedSha, Tag, object_class
from ..pack import (
from ..repo import Repo
class TestObject(cls):
    """Class that inherits from the given class, but without __slots__.

        Note that classes with __slots__ can't have arbitrary attributes
        monkey-patched in, so this is a class that is exactly the same only
        with a __dict__ instead of __slots__.
        """