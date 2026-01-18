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
def update_references(self, revid=None):
    if revid is None:
        revid = self.target.last_revision()
    tree = self.target.repository.revision_tree(revid)
    try:
        with tree.get_file('.gitmodules') as f:
            for path, url, section in parse_submodules(GitConfigFile.from_file(f)):
                self.target.set_reference_info(tree.path2id(decode_git_path(path)), url.decode('utf-8'), decode_git_path(path))
    except transport.NoSuchFile:
        pass