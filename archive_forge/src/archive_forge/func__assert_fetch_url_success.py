from __future__ import absolute_import, division, print_function
import copy
import datetime
import json
import locale
import time
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import PY3
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_openssl_cli import (
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
def _assert_fetch_url_success(module, response, info, allow_redirect=False, allow_client_error=True, allow_server_error=True):
    if info['status'] < 0:
        raise NetworkException(msg='Failure downloading %s, %s' % (info['url'], info['msg']))
    if 300 <= info['status'] < 400 and (not allow_redirect) or (400 <= info['status'] < 500 and (not allow_client_error)) or (info['status'] >= 500 and (not allow_server_error)):
        raise ACMEProtocolException(module, info=info, response=response)