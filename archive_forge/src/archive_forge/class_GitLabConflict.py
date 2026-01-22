import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
class GitLabConflict(errors.BzrError):
    _fmt = 'Conflict during operation: %(reason)s'

    def __init__(self, reason):
        errors.BzrError(self)
        self.reason = reason