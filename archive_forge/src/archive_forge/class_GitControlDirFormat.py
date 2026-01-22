import contextlib
import os
from dulwich.refs import SymrefLoop
from .. import branch as _mod_branch
from .. import errors as brz_errors
from .. import osutils, trace, urlutils
from ..controldir import (BranchReferenceLoop, ControlDir, ControlDirFormat,
from ..transport import (FileExists, NoSuchFile, do_catching_redirections,
from .mapping import decode_git_path, encode_git_path
from .push import GitPushResult
from .transportgit import OBJECTDIR, TransportObjectStore
class GitControlDirFormat(ControlDirFormat):
    colocated_branches = True
    fixed_components = True

    def __eq__(self, other):
        return type(self) == type(other)

    def is_supported(self):
        return True

    def network_name(self):
        return b'git'