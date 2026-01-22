from __future__ import absolute_import, division, print_function
import json
import os
import random
import string
import gzip
from io import BytesIO
from ansible.module_utils.urls import open_url
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import text_type
from ansible.module_utils.six.moves import http_client
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def _check_request_payload(self, req_pyld, cur_pyld, uri):
    """
        Checks the request payload with the values currently held by the
        service. Will check if changes are needed and if properties are
        supported by the service.

        :param req_pyld: dict containing the properties to apply
        :param cur_pyld: dict containing the properties currently set
        :param uri: string containing the URI being modified
        :return: dict containing response information
        """
    change_required = False
    for prop in req_pyld:
        if prop not in cur_pyld:
            return {'ret': False, 'changed': False, 'msg': '%s does not support the property %s' % (uri, prop), 'changes_required': False}
        if isinstance(req_pyld[prop], dict) and isinstance(cur_pyld[prop], dict):
            sub_resp = self._check_request_payload(req_pyld[prop], cur_pyld[prop], uri)
            if not sub_resp['ret']:
                return sub_resp
            if sub_resp['changes_required']:
                change_required = True
        elif req_pyld[prop] != cur_pyld[prop]:
            change_required = True
    resp = {'ret': True, 'changes_required': change_required}
    if not change_required:
        resp['changed'] = False
        resp['msg'] = 'Properties in %s are already set' % uri
    return resp