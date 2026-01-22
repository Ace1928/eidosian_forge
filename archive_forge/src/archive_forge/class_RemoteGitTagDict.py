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
class RemoteGitTagDict(GitTags):

    def set_tag(self, name, revid):
        sha = self.branch.lookup_bzr_revision_id(revid)[0]
        self._set_ref(name, sha)

    def delete_tag(self, name):
        self._set_ref(name, dulwich.client.ZERO_SHA)

    def _set_ref(self, name, sha):
        ref = tag_name_to_ref(name)

        def get_changed_refs(old_refs):
            ret = {}
            if sha == dulwich.client.ZERO_SHA and ref not in old_refs:
                raise NoSuchTag(name)
            ret[ref] = sha
            return ret

        def generate_pack_data(have, want, ofs_delta=False, progress=None):
            return pack_objects_to_data([])
        result = self.repository.send_pack(get_changed_refs, generate_pack_data)
        error = result.ref_status.get(ref)
        if error:
            raise error