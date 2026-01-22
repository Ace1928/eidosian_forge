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
def _resume_stream_with_vfs(self, response, src_format):
    """Resume sending a stream via VFS, first resending the record and
        substream that couldn't be sent via an insert_stream verb.
        """
    if response[0][0] == b'missing-basis':
        tokens, missing_keys = bencode.bdecode_as_tuple(response[0][1])
        tokens = [token.decode('utf-8') for token in tokens]
    else:
        tokens = []

    def resume_substream():
        yield from self._last_substream
        self._last_substream = None

    def resume_stream():
        yield ('inventory-deltas', resume_substream())
        yield from self._last_stream
    return self._insert_real(resume_stream(), src_format, tokens)