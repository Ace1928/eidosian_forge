import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple
import breezy
from .lazy_import import lazy_import
import stat
from breezy import (
from . import errors, mutabletree, osutils
from . import revision as _mod_revision
from .controldir import (ControlComponent, ControlComponentFormat,
from .i18n import gettext
from .symbol_versioning import deprecated_in, deprecated_method
from .trace import mutter, note
from .transport import NoSuchFile
def add_parent_tree_id(self, revision_id, allow_leftmost_as_ghost=False):
    """Add revision_id as a parent.

        This is equivalent to retrieving the current list of parent ids
        and setting the list to its value plus revision_id.

        Args:
          revision_id: The revision id to add to the parent list. It may
            be a ghost revision as long as its not the first parent to be
            added, or the allow_leftmost_as_ghost parameter is set True.
          allow_leftmost_as_ghost: Allow the first parent to be a ghost.
        """
    with self.lock_write():
        parents = self.get_parent_ids() + [revision_id]
        self.set_parent_ids(parents, allow_leftmost_as_ghost=len(parents) > 1 or allow_leftmost_as_ghost)