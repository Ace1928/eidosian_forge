from __future__ import absolute_import, division, print_function
import abc
import traceback
from ansible.module_utils import six
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.common import ArgumentSpec
from ansible_collections.community.crypto.plugins.module_utils.crypto.basic import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate_info import (
def _check_privatekey(self):
    """Check whether provided parameters match, assuming self.existing_certificate and self.privatekey have been populated."""
    if self.backend == 'cryptography':
        return cryptography_compare_public_keys(self.existing_certificate.public_key(), self.privatekey.public_key())