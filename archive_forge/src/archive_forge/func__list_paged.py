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
def _list_paged(self, path, parameters=None, per_page=None):
    if parameters is None:
        parameters = {}
    else:
        parameters = dict(parameters.items())
    if per_page:
        parameters['per_page'] = str(per_page)
    page = 1
    while path:
        parameters['page'] = str(page)
        response = self._api_request('GET', path + '?' + ';'.join(['{}={}'.format(k, urlutils.quote(v)) for k, v in parameters.items()]))
        if response.status != 200:
            raise UnexpectedHttpStatus(path, response.status, headers=response.getheaders())
        data = json.loads(response.text)
        if not data:
            break
        yield data
        page += 1