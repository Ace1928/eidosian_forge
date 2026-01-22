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
class RemoteBranchConfig(RemoteConfig):
    """A RemoteConfig for Branches."""

    def __init__(self, branch):
        self._branch = branch

    def _get_configobj(self):
        path = self._branch._remote_path()
        response = self._branch._client.call_expecting_body(b'Branch.get_config_file', path)
        return self._response_to_configobj(response)

    def set_option(self, value, name, section=None):
        """Set the value associated with a named option.

        :param value: The value to set
        :param name: The name of the value to set
        :param section: The section the option is in (if any)
        """
        medium = self._branch._client._medium
        if medium._is_remote_before((1, 14)):
            return self._vfs_set_option(value, name, section)
        if isinstance(value, dict):
            if medium._is_remote_before((2, 2)):
                return self._vfs_set_option(value, name, section)
            return self._set_config_option_dict(value, name, section)
        else:
            return self._set_config_option(value, name, section)

    def _set_config_option(self, value, name, section):
        if isinstance(value, (bool, int)):
            value = str(value)
        elif isinstance(value, str):
            pass
        else:
            raise TypeError(value)
        try:
            path = self._branch._remote_path()
            response = self._branch._client.call(b'Branch.set_config_option', path, self._branch._lock_token, self._branch._repo_lock_token, value.encode('utf-8'), name.encode('utf-8'), (section or '').encode('utf-8'))
        except errors.UnknownSmartMethod:
            medium = self._branch._client._medium
            medium._remember_remote_is_before((1, 14))
            return self._vfs_set_option(value, name, section)
        if response != ():
            raise errors.UnexpectedSmartServerResponse(response)

    def _serialize_option_dict(self, option_dict):
        utf8_dict = {}
        for key, value in option_dict.items():
            if isinstance(key, str):
                key = key.encode('utf8')
            if isinstance(value, str):
                value = value.encode('utf8')
            utf8_dict[key] = value
        return bencode.bencode(utf8_dict)

    def _set_config_option_dict(self, value, name, section):
        try:
            path = self._branch._remote_path()
            serialised_dict = self._serialize_option_dict(value)
            response = self._branch._client.call(b'Branch.set_config_option_dict', path, self._branch._lock_token, self._branch._repo_lock_token, serialised_dict, name.encode('utf-8'), (section or '').encode('utf-8'))
        except errors.UnknownSmartMethod:
            medium = self._branch._client._medium
            medium._remember_remote_is_before((2, 2))
            return self._vfs_set_option(value, name, section)
        if response != ():
            raise errors.UnexpectedSmartServerResponse(response)

    def _real_object(self):
        self._branch._ensure_real()
        return self._branch._real_branch

    def _vfs_set_option(self, value, name, section=None):
        return self._real_object()._get_config().set_option(value, name, section)