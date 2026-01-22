from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
class SwitchHookParams:
    """Object holding parameters passed to `*_switch` hooks.

    There are 4 fields that hooks may wish to access:

    Attributes:
      control_dir: ControlDir of the checkout to change
      to_branch: branch that the checkout is to reference
      force: skip the check for local commits in a heavy checkout
      revision_id: revision ID to switch to (or None)
    """

    def __init__(self, control_dir, to_branch, force, revision_id):
        """Create a group of SwitchHook parameters.

        Args:
          control_dir: ControlDir of the checkout to change
          to_branch: branch that the checkout is to reference
          force: skip the check for local commits in a heavy checkout
          revision_id: revision ID to switch to (or None)
        """
        self.control_dir = control_dir
        self.to_branch = to_branch
        self.force = force
        self.revision_id = revision_id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return '<{} for {} to ({}, {})>'.format(self.__class__.__name__, self.control_dir, self.to_branch, self.revision_id)