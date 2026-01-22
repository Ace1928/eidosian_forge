import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def _unique_lines(self, matching_blocks):
    """Analyse matching_blocks to determine which lines are unique

        :return: a tuple of (unique_left, unique_right), where the values are
            sets of line numbers of unique lines.
        """
    last_i = 0
    last_j = 0
    unique_left = []
    unique_right = []
    for i, j, n in matching_blocks:
        unique_left.extend(range(last_i, i))
        unique_right.extend(range(last_j, j))
        last_i = i + n
        last_j = j + n
    return (unique_left, unique_right)