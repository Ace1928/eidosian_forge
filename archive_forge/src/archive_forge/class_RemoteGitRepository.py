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
class RemoteGitRepository(GitRepository):
    supports_random_access = False

    @property
    def user_url(self):
        return self.control_url

    def get_parent_map(self, revids):
        raise GitSmartRemoteNotSupported(self.get_parent_map, self)

    def archive(self, *args, **kwargs):
        return self.controldir.archive(*args, **kwargs)

    def fetch_pack(self, determine_wants, graph_walker, pack_data, progress=None):
        return self.controldir.fetch_pack(determine_wants, graph_walker, pack_data, progress)

    def send_pack(self, get_changed_refs, generate_pack_data):
        return self.controldir.send_pack(get_changed_refs, generate_pack_data)

    def fetch_objects(self, determine_wants, graph_walker, resolve_ext_ref, progress=None):
        import tempfile
        fd, path = tempfile.mkstemp(suffix='.pack')
        try:
            self.fetch_pack(determine_wants, graph_walker, lambda x: os.write(fd, x), progress)
        finally:
            os.close(fd)
        if os.path.getsize(path) == 0:
            return EmptyObjectStoreIterator()
        return TemporaryPackIterator(path[:-len('.pack')], resolve_ext_ref)

    def lookup_bzr_revision_id(self, bzr_revid, mapping=None):
        try:
            return mapping_registry.revision_id_bzr_to_foreign(bzr_revid)
        except InvalidRevisionId:
            raise NoSuchRevision(self, bzr_revid)

    def lookup_foreign_revision_id(self, foreign_revid, mapping=None):
        """Lookup a revision id.

        """
        if mapping is None:
            mapping = self.get_mapping()
        return mapping.revision_id_foreign_to_bzr(foreign_revid)

    def revision_tree(self, revid):
        return GitRemoteRevisionTree(self, revid)

    def get_revisions(self, revids):
        raise GitSmartRemoteNotSupported(self.get_revisions, self)

    def has_revisions(self, revids):
        raise GitSmartRemoteNotSupported(self.get_revisions, self)