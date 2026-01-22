import logging
import os
import debtcollector.renames
from keystoneauth1 import access
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
from oslo_utils import importutils
import requests
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
def _authenticate_keystone(self):
    if self.user_id:
        creds = {'userId': self.user_id, 'password': self.password}
    else:
        creds = {'username': self.username, 'password': self.password}
    if self.project_id:
        body = {'auth': {'passwordCredentials': creds, 'tenantId': self.project_id}}
    else:
        body = {'auth': {'passwordCredentials': creds, 'tenantName': self.project_name}}
    if self.auth_url is None:
        raise exceptions.NoAuthURLProvided()
    token_url = self.auth_url + '/tokens'
    resp, resp_body = self._cs_request(token_url, 'POST', body=jsonutils.dumps(body), content_type='application/json', allow_redirects=True)
    if resp.status_code != 200:
        raise exceptions.Unauthorized(message=resp_body)
    if resp_body:
        try:
            resp_body = jsonutils.loads(resp_body)
        except ValueError:
            pass
    else:
        resp_body = None
    self._extract_service_catalog(resp_body)