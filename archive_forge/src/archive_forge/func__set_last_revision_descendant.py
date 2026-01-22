import bz2
import os
import re
import sys
import zlib
from typing import Callable, List, Optional
import fastbencode as bencode
from .. import branch
from .. import bzr as _mod_bzr
from .. import config as _mod_config
from .. import (controldir, debug, errors, gpg, graph, lock, lockdir, osutils,
from .. import repository as _mod_repository
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..branch import BranchWriteLockResult
from ..decorators import only_raises
from ..errors import NoSuchRevision, SmartProtocolError
from ..i18n import gettext
from ..lockable_files import LockableFiles
from ..repository import RepositoryWriteLockResult, _LazyListJoin
from ..revision import NULL_REVISION
from ..trace import log_exception_quietly, mutter, note, warning
from . import branch as bzrbranch
from . import bzrdir as _mod_bzrdir
from . import inventory_delta
from . import repository as bzrrepository
from . import testament as _mod_testament
from . import vf_repository, vf_search
from .branch import BranchReferenceFormat
from .inventory import Inventory
from .inventorytree import InventoryRevisionTree
from .serializer import format_registry as serializer_format_registry
from .smart import client
from .smart import repository as smart_repo
from .smart import vfs
from .smart.client import _SmartClient
from .versionedfile import FulltextContentFactory
def _set_last_revision_descendant(self, revision_id, other_branch, allow_diverged=False, allow_overwrite_descendant=False):
    old_revno, old_revid = self.last_revision_info()
    history = self._lefthand_history(revision_id)
    self._run_pre_change_branch_tip_hooks(len(history), revision_id)
    err_context = {'other_branch': other_branch}
    response = self._call(b'Branch.set_last_revision_ex', self._remote_path(), self._lock_token, self._repo_lock_token, revision_id, int(allow_diverged), int(allow_overwrite_descendant), **err_context)
    self._clear_cached_state()
    if len(response) != 3 and response[0] != b'ok':
        raise errors.UnexpectedSmartServerResponse(response)
    new_revno, new_revision_id = response[1:]
    self._last_revision_info_cache = (new_revno, new_revision_id)
    self._run_post_change_branch_tip_hooks(old_revno, old_revid)
    if self._real_branch is not None:
        cache = (new_revno, new_revision_id)
        self._real_branch._last_revision_info_cache = cache