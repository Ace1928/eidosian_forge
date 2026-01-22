import gzip
import re
from dulwich.refs import SymrefLoop
from .. import config, debug, errors, osutils, trace, ui, urlutils
from ..controldir import BranchReferenceLoop
from ..errors import (AlreadyBranchError, BzrError, ConnectionReset,
from ..push import PushResult
from ..revision import NULL_REVISION
from ..revisiontree import RevisionTree
from ..transport import (NoSuchFile, Transport,
from . import is_github_url, lazy_check_versions, user_agent_for_github
import os
import select
import urllib.parse as urlparse
import dulwich
import dulwich.client
from dulwich.errors import GitProtocolError, HangupException
from dulwich.pack import (PACK_SPOOL_FILE_MAX_SIZE, Pack, load_pack_index,
from dulwich.protocol import ZERO_SHA
from dulwich.refs import SYMREF, DictRefsContainer
from dulwich.repo import NotGitRepository
from .branch import (GitBranch, GitBranchFormat, GitBranchPushResult, GitTags,
from .dir import GitControlDirFormat, GitDir
from .errors import GitSmartRemoteNotSupported
from .mapping import encode_git_path, mapping_registry
from .object_store import get_object_store
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_peeled, ref_to_tag_name,
from .repository import GitRepository, GitRepositoryFormat
class RemoteGitControlDirFormat(GitControlDirFormat):
    """The .git directory control format."""
    supports_workingtrees = False

    @classmethod
    def _known_formats(self):
        return {RemoteGitControlDirFormat()}

    def get_branch_format(self):
        return RemoteGitBranchFormat()

    @property
    def repository_format(self):
        return GitRepositoryFormat()

    def is_initializable(self):
        return False

    def is_supported(self):
        return True

    def open(self, transport, _found=None):
        """Open this directory.

        """
        split_url = _git_url_and_path_from_transport(transport.external_url())
        if isinstance(transport, GitSmartTransport):
            client = transport._get_client()
        elif split_url.scheme in ('http', 'https'):
            client = BzrGitHttpClient(transport)
        elif split_url.scheme in ('file',):
            client = dulwich.client.LocalGitClient()
        else:
            raise NotBranchError(transport.base)
        if not _found:
            pass
        return RemoteGitDir(transport, self, client, split_url.path)

    def get_format_description(self):
        return 'Remote Git Repository'

    def initialize_on_transport(self, transport):
        raise UninitializableFormat(self)

    def supports_transport(self, transport):
        try:
            external_url = transport.external_url()
        except InProcessTransport:
            raise NotBranchError(path=transport.base)
        return external_url.startswith('http:') or external_url.startswith('https:') or external_url.startswith('git+') or external_url.startswith('git:')