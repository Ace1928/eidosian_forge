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
def _decode_retry(module, response, info, retry_count):
    if info['status'] not in RETRY_STATUS_CODES:
        return False
    if retry_count >= RETRY_COUNT:
        raise ACMEProtocolException(module, msg='Giving up after {retry} retries'.format(retry=RETRY_COUNT), info=info, response=response)
    try:
        retry_after = min(max(1, int(info.get('retry-after'))), 60)
    except (TypeError, ValueError) as dummy:
        retry_after = 10
    module.log('Retrieved a %s HTTP status on %s, retrying in %s seconds' % (format_http_status(info['status']), info['url'], retry_after))
    time.sleep(retry_after)
    return True