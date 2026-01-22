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
class RemoteRepositoryFormat(vf_repository.VersionedFileRepositoryFormat):
    """Format for repositories accessed over a _SmartClient.

    Instances of this repository are represented by RemoteRepository
    instances.

    The RemoteRepositoryFormat is parameterized during construction
    to reflect the capabilities of the real, remote format. Specifically
    the attributes rich_root_data and supports_tree_reference are set
    on a per instance basis, and are not set (and should not be) at
    the class level.

    :ivar _custom_format: If set, a specific concrete repository format that
        will be used when initializing a repository with this
        RemoteRepositoryFormat.
    :ivar _creating_repo: If set, the repository object that this
        RemoteRepositoryFormat was created for: it can be called into
        to obtain data like the network name.
    """
    _matchingcontroldir = RemoteBzrDirFormat()
    supports_full_versioned_files = True
    supports_leaving_lock = True
    supports_overriding_transport = False
    supports_ghosts = False

    def __init__(self):
        _mod_repository.RepositoryFormat.__init__(self)
        self._custom_format = None
        self._network_name = None
        self._creating_bzrdir = None
        self._revision_graph_can_have_wrong_parents = None
        self._supports_chks = None
        self._supports_external_lookups = None
        self._supports_tree_reference = None
        self._supports_funky_characters = None
        self._supports_nesting_repositories = None
        self._rich_root_data = None

    def __repr__(self):
        return '{}(_network_name={!r})'.format(self.__class__.__name__, self._network_name)

    @property
    def fast_deltas(self):
        self._ensure_real()
        return self._custom_format.fast_deltas

    @property
    def rich_root_data(self):
        if self._rich_root_data is None:
            self._ensure_real()
            self._rich_root_data = self._custom_format.rich_root_data
        return self._rich_root_data

    @property
    def supports_chks(self):
        if self._supports_chks is None:
            self._ensure_real()
            self._supports_chks = self._custom_format.supports_chks
        return self._supports_chks

    @property
    def supports_external_lookups(self):
        if self._supports_external_lookups is None:
            self._ensure_real()
            self._supports_external_lookups = self._custom_format.supports_external_lookups
        return self._supports_external_lookups

    @property
    def supports_funky_characters(self):
        if self._supports_funky_characters is None:
            self._ensure_real()
            self._supports_funky_characters = self._custom_format.supports_funky_characters
        return self._supports_funky_characters

    @property
    def supports_nesting_repositories(self):
        if self._supports_nesting_repositories is None:
            self._ensure_real()
            self._supports_nesting_repositories = self._custom_format.supports_nesting_repositories
        return self._supports_nesting_repositories

    @property
    def supports_tree_reference(self):
        if self._supports_tree_reference is None:
            self._ensure_real()
            self._supports_tree_reference = self._custom_format.supports_tree_reference
        return self._supports_tree_reference

    @property
    def revision_graph_can_have_wrong_parents(self):
        if self._revision_graph_can_have_wrong_parents is None:
            self._ensure_real()
            self._revision_graph_can_have_wrong_parents = self._custom_format.revision_graph_can_have_wrong_parents
        return self._revision_graph_can_have_wrong_parents

    def _vfs_initialize(self, a_controldir, shared):
        """Helper for common code in initialize."""
        if self._custom_format:
            result = self._custom_format.initialize(a_controldir, shared=shared)
        elif self._creating_bzrdir is not None:
            prior_repo = self._creating_bzrdir.open_repository()
            prior_repo._ensure_real()
            result = prior_repo._real_repository._format.initialize(a_controldir, shared=shared)
        else:
            a_controldir._ensure_real()
            result = a_controldir._real_bzrdir.create_repository(shared=shared)
        if not isinstance(result, RemoteRepository):
            return self.open(a_controldir)
        else:
            return result

    def initialize(self, a_controldir, shared=False):
        if not isinstance(a_controldir, RemoteBzrDir):
            return self._vfs_initialize(a_controldir, shared)
        medium = a_controldir._client._medium
        if medium._is_remote_before((1, 13)):
            return self._vfs_initialize(a_controldir, shared)
        if self._custom_format:
            network_name = self._custom_format.network_name()
        elif self._network_name:
            network_name = self._network_name
        else:
            reference_bzrdir_format = controldir.format_registry.get('default')()
            reference_format = reference_bzrdir_format.repository_format
            network_name = reference_format.network_name()
        path = a_controldir._path_for_remote_call(a_controldir._client)
        verb = b'BzrDir.create_repository'
        if shared:
            shared_str = b'True'
        else:
            shared_str = b'False'
        try:
            response = a_controldir._call(verb, path, network_name, shared_str)
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 13))
            return self._vfs_initialize(a_controldir, shared)
        else:
            format = response_tuple_to_repo_format(response[1:])
            format._creating_bzrdir = a_controldir
            remote_repo = RemoteRepository(a_controldir, format)
            format._creating_repo = remote_repo
            return remote_repo

    def open(self, a_controldir):
        if not isinstance(a_controldir, RemoteBzrDir):
            raise AssertionError('{!r} is not a RemoteBzrDir'.format(a_controldir))
        return a_controldir.open_repository()

    def _ensure_real(self):
        if self._custom_format is None:
            try:
                self._custom_format = _mod_repository.network_format_registry.get(self._network_name)
            except KeyError as e:
                raise errors.UnknownFormatError(kind='repository', format=self._network_name) from e

    @property
    def _fetch_order(self):
        self._ensure_real()
        return self._custom_format._fetch_order

    @property
    def _fetch_uses_deltas(self):
        self._ensure_real()
        return self._custom_format._fetch_uses_deltas

    @property
    def _fetch_reconcile(self):
        self._ensure_real()
        return self._custom_format._fetch_reconcile

    def get_format_description(self):
        self._ensure_real()
        return 'Remote: ' + self._custom_format.get_format_description()

    def __eq__(self, other):
        return self.__class__ is other.__class__

    def network_name(self):
        if self._network_name:
            return self._network_name
        self._creating_repo._ensure_real()
        return self._creating_repo._real_repository._format.network_name()

    @property
    def pack_compresses(self):
        self._ensure_real()
        return self._custom_format.pack_compresses

    @property
    def _serializer(self):
        self._ensure_real()
        return self._custom_format._serializer