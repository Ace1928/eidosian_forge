import re
import shutil
import tempfile
from typing import Any, List, Optional
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ...forge import (AutoMergeUnsupported, Forge, LabelsUnsupported,
from ...git.urls import git_url_to_bzr_url
from ...lazy_import import lazy_import
from ...trace import mutter
from breezy.plugins.launchpad import (
from ...transport import get_transport
def _publish_git(self, local_branch, base_path, name, owner, project=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
    if tag_selector is None:
        tag_selector = lambda t: False
    to_path = self._get_derived_git_path(base_path, owner, project)
    to_transport = get_transport(GIT_SCHEME_MAP['ssh'] + to_path)
    try:
        dir_to = controldir.ControlDir.open_from_transport(to_transport)
    except errors.NotBranchError:
        dir_to = None
    if dir_to is None:
        try:
            br_to = local_branch.create_clone_on_transport(to_transport, revision_id=revision_id, name=name, tag_selector=tag_selector)
        except errors.NoRoundtrippingSupport:
            br_to = local_branch.create_clone_on_transport(to_transport, revision_id=revision_id, name=name, lossy=True, tag_selector=tag_selector)
    else:
        try:
            dir_to = dir_to.push_branch(local_branch, revision_id, overwrite=overwrite, name=name, tag_selector=tag_selector)
        except errors.NoRoundtrippingSupport:
            if not allow_lossy:
                raise
            dir_to = dir_to.push_branch(local_branch, revision_id, overwrite=overwrite, name=name, lossy=True, tag_selector=tag_selector)
        br_to = dir_to.target_branch
    return (br_to, 'https://git.launchpad.net/{}/+ref/{}'.format(to_path, name))