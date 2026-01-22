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
def _check_extensions(csr):
    extensions = csr.extensions
    return _check_subjectAltName(extensions) and _check_keyUsage(extensions) and _check_extenededKeyUsage(extensions) and _check_basicConstraints(extensions) and _check_ocspMustStaple(extensions) and _check_subject_key_identifier(extensions) and _check_authority_key_identifier(extensions) and _check_nameConstraints(extensions) and _check_crl_distribution_points(extensions)