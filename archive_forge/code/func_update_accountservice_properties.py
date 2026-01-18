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
def update_accountservice_properties(self, user):
    account_properties = user.get('account_properties')
    if account_properties is None:
        return {'ret': False, 'msg': 'Must provide account_properties for UpdateAccountServiceProperties command'}
    response = self.get_request(self.root_uri + self.service_root)
    if response['ret'] is False:
        return response
    data = response['data']
    accountservice_uri = data.get('AccountService', {}).get('@odata.id')
    if accountservice_uri is None:
        return {'ret': False, 'msg': 'AccountService resource not found'}
    resp = self.patch_request(self.root_uri + accountservice_uri, account_properties, check_pyld=True)
    if resp['ret'] and resp['changed']:
        resp['msg'] = 'Modified account service'
    return resp