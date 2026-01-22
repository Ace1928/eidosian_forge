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
def _get_derived_git_path(self, base_path, owner, project):
    base_repo = self.launchpad.git_repositories.getByPath(path=base_path)
    if project is None:
        project = urlutils.parse_url(base_repo.git_ssh_url)[-1].strip('/')
    if project.startswith('~'):
        project = '/'.join(base_path.split('/')[1:])
    return '~{}/{}'.format(owner, project)