import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ... import version_string as breezy_version
from ...config import AuthenticationConfig, GlobalStack
from ...errors import (InvalidHttpResponse, PermissionDenied,
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...i18n import gettext
from ...trace import note
from ...transport import get_transport
from ...transport.http import default_user_agent
def _create_pull(self, path, title, head, base, body=None, labels=None, assignee=None, draft=False, maintainer_can_modify=False):
    data = {'title': title, 'head': head, 'base': base, 'draft': draft, 'maintainer_can_modify': maintainer_can_modify}
    if labels is not None:
        data['labels'] = labels
    if assignee is not None:
        data['assignee'] = assignee
    if body:
        data['body'] = body
    response = self._api_request('POST', path, body=json.dumps(data).encode('utf-8'))
    if response.status == 403:
        raise PermissionDenied(path, response.text)
    if response.status == 422:
        raise ValidationFailed(json.loads(response.text))
    if response.status != 201:
        raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
    return json.loads(response.text)