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
class RemoteBranchStore(_mod_config.IniFileStore):
    """Branch store which attempts to use HPSS calls to retrieve branch store.

    Note that this is specific to bzr-based formats.
    """

    def __init__(self, branch):
        super().__init__()
        self.branch = branch
        self.id = 'branch'
        self._real_store = None

    def external_url(self):
        return urlutils.join(self.branch.user_url, 'branch.conf')

    def _load_content(self):
        path = self.branch._remote_path()
        try:
            response, handler = self.branch._call_expecting_body(b'Branch.get_config_file', path)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_store._load_content()
        if len(response) and response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        return handler.read_body_bytes()

    def _save_content(self, content):
        path = self.branch._remote_path()
        try:
            response, handler = self.branch._call_with_body_bytes_expecting_body(b'Branch.put_config_file', (path, self.branch._lock_token, self.branch._repo_lock_token), content)
        except errors.UnknownSmartMethod:
            self._ensure_real()
            return self._real_store._save_content(content)
        handler.cancel_read_body()
        if response != (b'ok',):
            raise errors.UnexpectedSmartServerResponse(response)

    def _ensure_real(self):
        self.branch._ensure_real()
        if self._real_store is None:
            self._real_store = _mod_config.BranchStore(self.branch)