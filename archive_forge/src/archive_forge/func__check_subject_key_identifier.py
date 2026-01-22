from __future__ import absolute_import, division, print_function
import abc
import binascii
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_crl import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.csr_info import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
def _check_subject_key_identifier(extensions):
    ext = _find_extension(extensions, cryptography.x509.SubjectKeyIdentifier)
    if self.create_subject_key_identifier or self.subject_key_identifier is not None:
        if not ext or ext.critical:
            return False
        if self.create_subject_key_identifier:
            digest = cryptography.x509.SubjectKeyIdentifier.from_public_key(self.privatekey.public_key()).digest
            return ext.value.digest == digest
        else:
            return ext.value.digest == self.subject_key_identifier
    else:
        return ext is None