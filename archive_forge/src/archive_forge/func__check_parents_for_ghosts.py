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
def _check_parents_for_ghosts(self, revision_ids, allow_leftmost_as_ghost):
    """Common ghost checking functionality from set_parent_*.

        This checks that the left hand-parent exists if there are any
        revisions present.
        """
    if len(revision_ids) > 0:
        leftmost_id = revision_ids[0]
        if not allow_leftmost_as_ghost and (not self.branch.repository.has_revision(leftmost_id)):
            raise errors.GhostRevisionUnusableHere(leftmost_id)