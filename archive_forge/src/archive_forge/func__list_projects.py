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
def _list_projects(self, owner):
    path = 'users/%s/projects' % urlutils.quote(str(owner), '')
    parameters = {}
    return self._list_paged(path, parameters, per_page=DEFAULT_PAGE_SIZE)