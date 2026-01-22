from __future__ import absolute_import, division, print_function
import base64
import binascii
import datetime
import os
import traceback
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.acme.backends import (
from ansible_collections.community.crypto.plugins.module_utils.acme.certificates import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.io import read_file
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import nopad_b64
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
def _match_subject(self, x509_subject, match_subject):
    for oid, value in match_subject:
        found = False
        for attribute in x509_subject:
            if attribute.oid == oid and value == to_native(attribute.value):
                found = True
                break
        if not found:
            return False
    return True