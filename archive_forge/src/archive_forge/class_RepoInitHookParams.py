from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Type, cast
from .lazy_import import lazy_import
import textwrap
from breezy import (
from breezy.i18n import gettext
from . import errors, hooks, registry
from . import revision as _mod_revision
from . import trace
from . import transport as _mod_transport
class RepoInitHookParams:
    """Object holding parameters passed to ``*_repo_init`` hooks.

    There are 4 fields that hooks may wish to access:

    Attributes:
      repository: Repository created
      format: Repository format
      bzrdir: The controldir for the repository
      shared: The repository is shared
    """

    def __init__(self, repository, format, controldir, shared):
        """Create a group of RepoInitHook parameters.

        Args:
          repository: Repository created
          format: Repository format
          controldir: The controldir for the repository
          shared: The repository is shared
        """
        self.repository = repository
        self.format = format
        self.controldir = controldir
        self.shared = shared

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        if self.repository:
            return '<{} for {}>'.format(self.__class__.__name__, self.repository)
        else:
            return '<{} for {}>'.format(self.__class__.__name__, self.controldir)