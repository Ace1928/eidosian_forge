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
class _InventoryNoneEntry:
    """This represents an inventory entry which *isn't there*.

    It simplifies the merging logic if we always have an InventoryEntry, even
    if it isn't actually present
    """
    executable = None
    kind = None
    name = None
    parent_id = None
    revision = None
    symlink_target = None
    text_sha1 = None

    def is_unmodified(self, other):
        return other is self