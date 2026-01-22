from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
def _add_fallback(self, repository, possible_transports=None):
    """Add a fallback to the supplied repository, if stacking is set."""
    stack_on = self._get_full_stack_on()
    if stack_on is None:
        return
    try:
        stacked_dir = ControlDir.open(stack_on, possible_transports=possible_transports)
    except errors.JailBreak:
        return
    try:
        stacked_repo = stacked_dir.open_branch().repository
    except errors.NotBranchError:
        stacked_repo = stacked_dir.open_repository()
    try:
        repository.add_fallback_repository(stacked_repo)
    except errors.UnstackableRepositoryFormat:
        if self._require_stacking:
            raise
    else:
        self._require_stacking = True