from __future__ import absolute_import, division, print_function
import datetime
import time
import os
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible_collections.community.crypto.plugins.module_utils.ecs.api import ECSClient, RestOperationException, SessionConfigurationException
from ansible_collections.community.crypto.plugins.module_utils.crypto.support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.module_backends.certificate import (
class EntrustCertificateProvider(CertificateProvider):

    def validate_module_args(self, module):
        pass

    def needs_version_two_certs(self, module):
        return False

    def create_backend(self, module, backend):
        return EntrustCertificateBackend(module, backend)