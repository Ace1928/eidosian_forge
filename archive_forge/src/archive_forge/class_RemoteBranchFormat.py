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
class RemoteBranchFormat(branch.BranchFormat):

    def __init__(self, network_name=None):
        super().__init__()
        self._matchingcontroldir = RemoteBzrDirFormat()
        self._matchingcontroldir.set_branch_format(self)
        self._custom_format = None
        self._network_name = network_name

    def __eq__(self, other):
        return isinstance(other, RemoteBranchFormat) and self.__dict__ == other.__dict__

    def _ensure_real(self):
        if self._custom_format is None:
            try:
                self._custom_format = branch.network_format_registry.get(self._network_name)
            except KeyError:
                raise errors.UnknownFormatError(kind='branch', format=self._network_name)

    def get_format_description(self):
        self._ensure_real()
        return 'Remote: ' + self._custom_format.get_format_description()

    def network_name(self):
        return self._network_name

    def open(self, a_controldir, name=None, ignore_fallbacks=False):
        return a_controldir.open_branch(name=name, ignore_fallbacks=ignore_fallbacks)

    def _vfs_initialize(self, a_controldir, name, append_revisions_only, repository=None):
        if isinstance(a_controldir, RemoteBzrDir):
            a_controldir._ensure_real()
            result = self._custom_format.initialize(a_controldir._real_bzrdir, name=name, append_revisions_only=append_revisions_only, repository=repository)
        else:
            result = self._custom_format.initialize(a_controldir, name=name, append_revisions_only=append_revisions_only, repository=repository)
        if isinstance(a_controldir, RemoteBzrDir) and (not isinstance(result, RemoteBranch)):
            result = RemoteBranch(a_controldir, a_controldir.find_repository(), result, name=name)
        return result

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        if name is None:
            name = a_controldir._get_selected_branch()
        if self._custom_format:
            network_name = self._custom_format.network_name()
        else:
            reference_bzrdir_format = controldir.format_registry.get('default')()
            reference_format = reference_bzrdir_format.get_branch_format()
            self._custom_format = reference_format
            network_name = reference_format.network_name()
        if not isinstance(a_controldir, RemoteBzrDir):
            return self._vfs_initialize(a_controldir, name=name, append_revisions_only=append_revisions_only, repository=repository)
        medium = a_controldir._client._medium
        if medium._is_remote_before((1, 13)):
            return self._vfs_initialize(a_controldir, name=name, append_revisions_only=append_revisions_only, repository=repository)
        path = a_controldir._path_for_remote_call(a_controldir._client)
        if name != '':
            raise controldir.NoColocatedBranchSupport(self)
        verb = b'BzrDir.create_branch'
        try:
            response = a_controldir._call(verb, path, network_name)
        except errors.UnknownSmartMethod:
            medium._remember_remote_is_before((1, 13))
            return self._vfs_initialize(a_controldir, name=name, append_revisions_only=append_revisions_only, repository=repository)
        if response[0] != b'ok':
            raise errors.UnexpectedSmartServerResponse(response)
        format = RemoteBranchFormat(network_name=response[1])
        repo_format = response_tuple_to_repo_format(response[3:])
        repo_path = response[2].decode('utf-8')
        if repository is not None:
            remote_repo_url = urlutils.join(a_controldir.user_url, repo_path)
            url_diff = urlutils.relative_url(repository.user_url, remote_repo_url)
            if url_diff != '.':
                raise AssertionError('repository.user_url %r does not match URL from server response (%r + %r)' % (repository.user_url, a_controldir.user_url, repo_path))
            remote_repo = repository
        else:
            if repo_path == '':
                repo_bzrdir = a_controldir
            else:
                repo_bzrdir = RemoteBzrDir(a_controldir.root_transport.clone(repo_path), a_controldir._format, a_controldir._client)
            remote_repo = RemoteRepository(repo_bzrdir, repo_format)
        remote_branch = RemoteBranch(a_controldir, remote_repo, format=format, setup_stacking=False, name=name)
        if append_revisions_only:
            remote_branch.set_append_revisions_only(append_revisions_only)
        remote_branch._last_revision_info_cache = (0, NULL_REVISION)
        return remote_branch

    def make_tags(self, branch):
        self._ensure_real()
        return self._custom_format.make_tags(branch)

    def supports_tags(self):
        self._ensure_real()
        return self._custom_format.supports_tags()

    def supports_stacking(self):
        self._ensure_real()
        return self._custom_format.supports_stacking()

    def supports_set_append_revisions_only(self):
        self._ensure_real()
        return self._custom_format.supports_set_append_revisions_only()

    @property
    def supports_reference_locations(self):
        self._ensure_real()
        return self._custom_format.supports_reference_locations

    def stores_revno(self):
        return True

    def _use_default_local_heads_to_fetch(self):
        self._ensure_real()
        if isinstance(self._custom_format, bzrbranch.BranchFormatMetadir):
            branch_class = self._custom_format._branch_class()
            heads_to_fetch_impl = branch_class.heads_to_fetch
            if heads_to_fetch_impl is branch.Branch.heads_to_fetch:
                return True
        return False