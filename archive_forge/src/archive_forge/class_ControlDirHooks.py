from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class ControlDirHooks(hooks.Hooks):
    """Hooks for ControlDir operations."""

    def __init__(self):
        """Create the default hooks."""
        hooks.Hooks.__init__(self, 'breezy.controldir', 'ControlDir.hooks')
        self.add_hook('pre_open', 'Invoked before attempting to open a ControlDir with the transport that the open will use.', (1, 14))
        self.add_hook('post_repo_init', 'Invoked after a repository has been initialized. post_repo_init is called with a breezy.controldir.RepoInitHookParams.', (2, 2))