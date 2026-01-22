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
class GitLabError(errors.BzrError):
    _fmt = 'GitLab error: %(error)s'

    def __init__(self, error, full_response):
        errors.BzrError.__init__(self)
        self.error = error
        self.full_response = full_response