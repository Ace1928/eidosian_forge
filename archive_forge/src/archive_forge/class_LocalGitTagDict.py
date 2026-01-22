import contextlib
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Dict, Optional, Set, Tuple
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.objects import ZERO_SHA, NotCommitError
from dulwich.repo import check_ref_format
from .. import branch, config, controldir, errors, lock
from .. import repository as _mod_repository
from .. import revision, trace, transport, urlutils
from ..foreign import ForeignBranch
from ..revision import NULL_REVISION
from ..tag import InterTags, TagConflict, Tags, TagSelector, TagUpdates
from ..trace import is_quiet, mutter, warning
from .errors import NoPushSupport
from .mapping import decode_git_path, encode_git_path
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_tag, ref_to_branch_name,
from .unpeel_map import UnpeelMap
from .urls import bzr_url_to_git_url, git_url_to_bzr_url
class LocalGitTagDict(GitTags):
    """Dictionary with tags in a local repository."""

    def __init__(self, branch):
        super().__init__(branch)
        self.refs = self.repository.controldir._git.refs

    def _set_tag_dict(self, to_dict):
        extra = set(self.refs.allkeys())
        for k, revid in to_dict.items():
            name = tag_name_to_ref(k)
            if name in extra:
                extra.remove(name)
            try:
                self.set_tag(k, revid)
            except errors.GhostTagsNotSupported:
                pass
        for name in extra:
            if is_tag(name):
                del self.repository._git[name]

    def set_tag(self, name, revid):
        try:
            git_sha, mapping = self.branch.lookup_bzr_revision_id(revid)
        except errors.NoSuchRevision:
            raise errors.GhostTagsNotSupported(self)
        self.refs[tag_name_to_ref(name)] = git_sha
        self.branch._tag_refs = None

    def delete_tag(self, name):
        ref = tag_name_to_ref(name)
        if ref not in self.refs:
            raise errors.NoSuchTag(name)
        del self.refs[ref]
        self.branch._tag_refs = None