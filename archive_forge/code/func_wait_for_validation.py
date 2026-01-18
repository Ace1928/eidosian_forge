from __future__ import absolute_import, division, print_function
import base64
import hashlib
import json
import re
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
def wait_for_validation(self, client, callenge_type):
    while True:
        self.refresh(client)
        if self.status in ['valid', 'invalid', 'revoked']:
            break
        time.sleep(2)
    if self.status == 'invalid':
        self.raise_error('Status is "invalid"', module=client.module)
    return self.status == 'valid'