import errno
import os
import shutil
from contextlib import ExitStack
from typing import List, Optional
from .clean_tree import iter_deletables
from .errors import BzrError, DependencyNotPresent
from .osutils import is_inside
from .trace import warning
from .transform import revert
from .transport import NoSuchFile
from .tree import Tree
from .workingtree import WorkingTree
Create a commit.

        See WorkingTree.commit() for documentation.
        