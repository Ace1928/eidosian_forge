from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
def _create_certificate_auth_handler(self):
    try:
        context = ssl.create_default_context()
    except AttributeError as exc:
        self._fail_with_exc_info('SSL certificate authentication requires python 2.7 or later.', exc)
    if not self.validate_certs:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    try:
        context.load_cert_chain(self.cert_filepath, keyfile=self.key_filepath)
    except IOError as exc:
        self._fail_with_exc_info('Cannot load SSL certificate, check files exist.', exc)
    return zapi.urllib.request.HTTPSHandler(context=context)