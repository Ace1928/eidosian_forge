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
def _check_subjectAltName(extensions):
    current_altnames_ext = _find_extension(extensions, cryptography.x509.SubjectAlternativeName)
    current_altnames = [to_text(altname) for altname in current_altnames_ext.value] if current_altnames_ext else []
    altnames = [to_text(cryptography_get_name(altname)) for altname in self.subjectAltName] if self.subjectAltName else []
    if set(altnames) != set(current_altnames):
        return False
    if altnames:
        if current_altnames_ext.critical != self.subjectAltName_critical:
            return False
    return True