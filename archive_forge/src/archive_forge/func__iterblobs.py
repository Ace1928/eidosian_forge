import stat
from dulwich.index import commit_tree, read_submodule_head
from dulwich.objects import Blob, Commit
from .. import bugtracker
from .. import config as _mod_config
from .. import gpg, osutils
from .. import revision as _mod_revision
from .. import trace
from ..errors import BzrError, RootMissing, UnsupportedOperation
from ..repository import CommitBuilder
from .mapping import encode_git_path, fix_person_identifier, object_mode
from .tree import entry_factory
def _iterblobs(self):
    return ((path, sha, mode) for path, (mode, sha) in self._blobs.items())