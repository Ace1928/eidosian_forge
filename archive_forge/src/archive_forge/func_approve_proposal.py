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
def approve_proposal(self, mp):
    with self.source_branch.lock_read():
        _call_webservice(mp.createComment, vote='Approve', subject='', content='Rubberstamp! Proposer approves of own proposal.')
        _call_webservice(mp.setStatus, status='Approved', revid=self.source_branch.last_revision())