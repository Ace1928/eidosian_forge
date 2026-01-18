from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
def tmos_version(client):
    uri = 'https://{0}:{1}/mgmt/tm/sys/'.format(client.provider['server'], client.provider['server_port'])
    resp = client.api.get(uri)
    try:
        response = resp.json()
    except ValueError as ex:
        raise F5ModuleError(str(ex))
    if 'code' in response and response['code'] in [400, 403]:
        if 'message' in response:
            raise F5ModuleError(response['message'])
        else:
            raise F5ModuleError(resp.content)
    to_parse = urlparse(response['selfLink'])
    query = to_parse.query
    version = query.split('=')[1]
    return version