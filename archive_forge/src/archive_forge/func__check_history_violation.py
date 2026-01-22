from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from .. import errors, lockable_files
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import urlutils
from ..branch import (Branch, BranchFormat, BranchWriteLockResult,
from ..controldir import ControlDir
from ..decorators import only_raises
from ..lock import LogicalLockResult, _RelockDebugMixin
from ..trace import mutter
from . import bzrdir, rio
from .repository import MetaDirRepository
def _check_history_violation(self, revision_id):
    last_revision = self.last_revision()
    if _mod_revision.is_null(last_revision):
        return
    graph = self.repository.get_graph()
    for lh_ancestor in graph.iter_lefthand_ancestry(revision_id):
        if lh_ancestor == last_revision:
            return
    raise errors.AppendRevisionsOnlyViolation(self.user_url)