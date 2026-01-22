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
class RemoteConfig:
    """A Config that reads and writes from smart verbs.

    It is a low-level object that considers config data to be name/value pairs
    that may be associated with a section. Assigning meaning to the these
    values is done at higher levels like breezy.config.TreeConfig.
    """

    def get_option(self, name, section=None, default=None):
        """Return the value associated with a named option.

        :param name: The name of the value
        :param section: The section the option is in (if any)
        :param default: The value to return if the value is not set
        :return: The value or default value
        """
        try:
            configobj = self._get_configobj()
            section_obj = None
            if section is None:
                section_obj = configobj
            else:
                try:
                    section_obj = configobj[section]
                except KeyError:
                    pass
            if section_obj is None:
                value = default
            else:
                value = section_obj.get(name, default)
        except errors.UnknownSmartMethod:
            value = self._vfs_get_option(name, section, default)
        for hook in _mod_config.OldConfigHooks['get']:
            hook(self, name, value)
        return value

    def _response_to_configobj(self, response):
        if len(response[0]) and response[0][0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        lines = response[1].read_body_bytes().splitlines()
        conf = _mod_config.ConfigObj(lines, encoding='utf-8')
        for hook in _mod_config.OldConfigHooks['load']:
            hook(self)
        return conf