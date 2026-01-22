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
def _get_parent_map_rpc(self, keys):
    """Helper for get_parent_map that performs the RPC."""
    medium = self._client._medium
    if medium._is_remote_before((1, 2)):
        rg = self._get_revision_graph(None)
        for node_id, parent_ids in rg.items():
            if parent_ids == ():
                rg[node_id] = (NULL_REVISION,)
        rg[NULL_REVISION] = ()
        return rg
    keys = set(keys)
    if None in keys:
        raise ValueError('get_parent_map(None) is not valid')
    if NULL_REVISION in keys:
        keys.discard(NULL_REVISION)
        found_parents = {NULL_REVISION: ()}
        if not keys:
            return found_parents
    else:
        found_parents = {}
    parents_map = self._unstacked_provider.get_cached_map()
    if parents_map is None:
        parents_map = {}
    if _DEFAULT_SEARCH_DEPTH <= 0:
        start_set, stop_keys, key_count = vf_search.search_result_from_parent_map(parents_map, self._unstacked_provider.missing_keys)
    else:
        start_set, stop_keys, key_count = vf_search.limited_search_result_from_parent_map(parents_map, self._unstacked_provider.missing_keys, keys, depth=_DEFAULT_SEARCH_DEPTH)
    recipe = ('manual', start_set, stop_keys, key_count)
    body = self._serialise_search_recipe(recipe)
    path = self.controldir._path_for_remote_call(self._client)
    for key in keys:
        if not isinstance(key, bytes):
            raise ValueError('key {!r} not a bytes string'.format(key))
    verb = b'Repository.get_parent_map'
    args = (path, b'include-missing:') + tuple(keys)
    try:
        response = self._call_with_body_bytes_expecting_body(verb, args, body)
    except errors.UnknownSmartMethod:
        warning('Server is too old for fast get_parent_map, reconnecting.  (Upgrade the server to Bazaar 1.2 to avoid this)')
        medium.disconnect()
        medium._remember_remote_is_before((1, 2))
        return self._get_parent_map_rpc(keys)
    response_tuple, response_handler = response
    if response_tuple[0] not in [b'ok']:
        response_handler.cancel_read_body()
        raise errors.UnexpectedSmartServerResponse(response_tuple)
    if response_tuple[0] == b'ok':
        coded = bz2.decompress(response_handler.read_body_bytes())
        if coded == b'':
            return {}
        lines = coded.split(b'\n')
        revision_graph = {}
        for line in lines:
            d = tuple(line.split())
            if len(d) > 1:
                revision_graph[d[0]] = d[1:]
            elif d[0].startswith(b'missing:'):
                revid = d[0][8:]
                self._unstacked_provider.note_missing_key(revid)
            else:
                revision_graph[d[0]] = (NULL_REVISION,)
        return revision_graph