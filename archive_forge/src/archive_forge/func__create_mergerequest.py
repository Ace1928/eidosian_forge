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
def _create_mergerequest(self, title, source_project_id, target_project_id, source_branch_name, target_branch_name, description, labels=None, allow_collaboration=False):
    path = 'projects/%s/merge_requests' % source_project_id
    fields = {'title': title, 'source_branch': source_branch_name, 'target_branch': target_branch_name, 'target_project_id': target_project_id, 'description': description, 'allow_collaboration': allow_collaboration}
    if labels:
        fields['labels'] = labels
    response = self._api_request('POST', path, fields=fields)
    if response.status == 400:
        data = json.loads(response.data)
        raise GitLabError(data.get('message'), response)
    if response.status == 403:
        raise errors.PermissionDenied(response.text)
    if response.status == 409:
        raise GitLabConflict(json.loads(response.data).get('message'))
    if response.status == 422:
        data = json.loads(response.data)
        raise GitLabUnprocessable(data.get('error') or data.get('message'), data)
    if response.status != 201:
        _unexpected_status(path, response)
    return json.loads(response.data)