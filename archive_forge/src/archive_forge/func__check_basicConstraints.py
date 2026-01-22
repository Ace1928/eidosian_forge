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
def _check_basicConstraints(extensions):
    bc_ext = _find_extension(extensions, cryptography.x509.BasicConstraints)
    current_ca = bc_ext.value.ca if bc_ext else False
    current_path_length = bc_ext.value.path_length if bc_ext else None
    ca, path_length = cryptography_get_basic_constraints(self.basicConstraints)
    if ca != current_ca:
        return False
    if path_length != current_path_length:
        return False
    if self.basicConstraints:
        return bc_ext is not None and bc_ext.critical == self.basicConstraints_critical
    else:
        return bc_ext is None