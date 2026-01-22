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
def _unstack(self):
    """Change a branch to be unstacked, copying data as needed.

        Don't call this directly, use set_stacked_on_url(None).
        """
    with ui.ui_factory.nested_progress_bar() as pb:
        old_repository = self.repository
        if len(old_repository._fallback_repositories) != 1:
            raise AssertionError("can't cope with fallback repositories of %r (fallbacks: %r)" % (old_repository, old_repository._fallback_repositories))
        new_bzrdir = controldir.ControlDir.open(self.controldir.root_transport.base)
        new_repository = new_bzrdir.find_repository()
        if new_repository._fallback_repositories:
            raise AssertionError("didn't expect %r to have fallback_repositories" % (self.repository,))
        lock_token = old_repository.lock_write().repository_token
        self.repository = new_repository
        if self._real_branch is not None:
            self._real_branch.repository = new_repository
        self.repository.lock_write(token=lock_token)
        if lock_token is not None:
            old_repository.leave_lock_in_place()
        old_repository.unlock()
        if lock_token is not None:
            self.repository.dont_leave_lock_in_place()
        old_lock_count = 0
        while True:
            try:
                old_repository.unlock()
            except errors.LockNotHeld:
                break
            old_lock_count += 1
        if old_lock_count == 0:
            raise AssertionError('old_repository should have been locked at least once.')
        for i in range(old_lock_count - 1):
            self.repository.lock_write()
        with old_repository.lock_read():
            try:
                tags_to_fetch = set(self.tags.get_reverse_tag_dict())
            except errors.TagsNotSupported:
                tags_to_fetch = set()
            fetch_spec = vf_search.NotInOtherForRevs(self.repository, old_repository, required_ids=[self.last_revision()], if_present_ids=tags_to_fetch, find_ghosts=True).execute()
            self.repository.fetch(old_repository, fetch_spec=fetch_spec)