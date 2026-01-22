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
def _vfs_initialize(self, a_controldir, name, append_revisions_only, repository=None):
    if isinstance(a_controldir, RemoteBzrDir):
        a_controldir._ensure_real()
        result = self._custom_format.initialize(a_controldir._real_bzrdir, name=name, append_revisions_only=append_revisions_only, repository=repository)
    else:
        result = self._custom_format.initialize(a_controldir, name=name, append_revisions_only=append_revisions_only, repository=repository)
    if isinstance(a_controldir, RemoteBzrDir) and (not isinstance(result, RemoteBranch)):
        result = RemoteBranch(a_controldir, a_controldir.find_repository(), result, name=name)
    return result