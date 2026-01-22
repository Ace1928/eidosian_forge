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
def _find_accountservice_resource(self):
    response = self.get_request(self.root_uri + self.service_root)
    if response['ret'] is False:
        return response
    data = response['data']
    if 'AccountService' not in data:
        return {'ret': False, 'msg': 'AccountService resource not found'}
    else:
        account_service = data['AccountService']['@odata.id']
        response = self.get_request(self.root_uri + account_service)
        if response['ret'] is False:
            return response
        data = response['data']
        accounts = data['Accounts']['@odata.id']
        if accounts[-1:] == '/':
            accounts = accounts[:-1]
        self.accounts_uri = accounts
    return {'ret': True}