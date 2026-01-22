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
def _get_endpoint_url(self):
    if self.auth_url is None:
        raise exceptions.NoAuthURLProvided()
    url = self.auth_url + '/tokens/%s/endpoints' % self.auth_token
    try:
        resp, body = self._cs_request(url, 'GET')
    except exceptions.Unauthorized:
        self.authenticate()
        return self.endpoint_url
    body = jsonutils.loads(body)
    for endpoint in body.get('endpoints', []):
        if endpoint['type'] == 'network' and endpoint.get('region') == self.region_name:
            if self.endpoint_type not in endpoint:
                raise exceptions.EndpointTypeNotFound(type_=self.endpoint_type)
            return endpoint[self.endpoint_type]
    raise exceptions.EndpointNotFound()