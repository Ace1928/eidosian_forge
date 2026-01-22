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
def _get_lp_git_ref_from_branch(self, branch):
    url, params = urlutils.split_segment_parameters(branch.user_url)
    scheme, user, password, host, port, path = urlutils.parse_url(url)
    repo_lp = self.launchpad.git_repositories.getByPath(path=path.strip('/'))
    try:
        ref_path = params['ref']
    except KeyError:
        branch_name = params.get('branch', branch.name)
        if branch_name:
            ref_path = 'refs/heads/%s' % branch_name
        else:
            ref_path = repo_lp.default_branch
    ref_lp = repo_lp.getRefByPath(path=ref_path)
    return (repo_lp, ref_lp)