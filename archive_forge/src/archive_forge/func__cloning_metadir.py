import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def _cloning_metadir(self):
    """Produce a metadir suitable for cloning with.

        :returns: (destination_bzrdir_format, source_repository)
        """
    result_format = self._format.__class__()
    try:
        try:
            branch = self.open_branch(ignore_fallbacks=True)
            source_repository = branch.repository
            result_format._branch_format = branch._format
        except errors.NotBranchError:
            source_repository = self.open_repository()
    except errors.NoRepositoryPresent:
        source_repository = None
    else:
        repo_format = source_repository._format
        if isinstance(repo_format, remote.RemoteRepositoryFormat):
            source_repository._ensure_real()
            repo_format = source_repository._real_repository._format
        result_format.repository_format = repo_format
    try:
        tree = self.open_workingtree(recommend_upgrade=False)
    except (errors.NoWorkingTree, errors.NotLocalUrl):
        result_format.workingtree_format = None
    else:
        result_format.workingtree_format = tree._format.__class__()
    return (result_format, source_repository)