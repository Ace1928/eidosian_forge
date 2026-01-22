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
class LocalGitBranchFormat(GitBranchFormat):

    def get_format_description(self):
        return 'Local Git Branch'

    @property
    def _matchingcontroldir(self):
        from .dir import LocalGitControlDirFormat
        return LocalGitControlDirFormat()

    def initialize(self, a_controldir, name=None, repository=None, append_revisions_only=None):
        from .dir import LocalGitDir
        if not isinstance(a_controldir, LocalGitDir):
            raise errors.IncompatibleFormat(self, a_controldir._format)
        return a_controldir.create_branch(repository=repository, name=name, append_revisions_only=append_revisions_only)