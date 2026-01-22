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
def _get_selected_ref(self, branch, ref=None):
    if ref is not None and branch is not None:
        raise brz_errors.BzrError("can't specify both ref and branch")
    if ref is not None:
        return ref
    if branch is not None:
        from .refs import branch_name_to_ref
        return branch_name_to_ref(branch)
    segment_parameters = getattr(self.user_transport, 'get_segment_parameters', lambda: {})()
    ref = segment_parameters.get('ref')
    if ref is not None:
        return urlutils.unquote_to_bytes(ref)
    if branch is None and getattr(self, '_get_selected_branch', False):
        branch = self._get_selected_branch()
        if branch is not None:
            from .refs import branch_name_to_ref
            return branch_name_to_ref(branch)
    return b'HEAD'