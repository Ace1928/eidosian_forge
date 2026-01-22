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
def _retrieve_user(self):
    if self._current_user:
        return
    try:
        response = self._api_request('GET', 'user')
    except errors.UnexpectedHttpStatus as e:
        if e.code == 401:
            raise GitLabLoginMissing(self.base_url)
        raise
    if response.status == 200:
        self._current_user = json.loads(response.data)
        return
    if response.status == 401:
        if json.loads(response.data) == {'message': '401 Unauthorized'}:
            raise GitLabLoginMissing(self.base_url)
        else:
            raise GitlabLoginError(response.text)
    raise UnsupportedForge(self.base_url)