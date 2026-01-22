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
class BranchInitHookParams:
    """Object holding parameters passed to `*_branch_init` hooks.

    There are 4 fields that hooks may wish to access:

    Attributes:
      format: the branch format
      bzrdir: the ControlDir where the branch will be/has been initialized
      name: name of colocated branch, if any (or None)
      branch: the branch created

    Note that for lightweight checkouts, the bzrdir and format fields refer to
    the checkout, hence they are different from the corresponding fields in
    branch, which refer to the original branch.
    """

    def __init__(self, format, controldir, name, branch):
        """Create a group of BranchInitHook parameters.

        Args:
          format: the branch format
          controldir: the ControlDir where the branch will be/has been
            initialized
          name: name of colocated branch, if any (or None)
          branch: the branch created

        Note that for lightweight checkouts, the bzrdir and format fields refer
        to the checkout, hence they are different from the corresponding fields
        in branch, which refer to the original branch.
        """
        self.format = format
        self.controldir = controldir
        self.name = name
        self.branch = branch

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return '<{} of {}>'.format(self.__class__.__name__, self.branch)