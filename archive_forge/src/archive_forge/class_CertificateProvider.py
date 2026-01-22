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
@six.add_metaclass(abc.ABCMeta)
class CertificateProvider(object):

    @abc.abstractmethod
    def validate_module_args(self, module):
        """Check module arguments"""

    @abc.abstractmethod
    def needs_version_two_certs(self, module):
        """Whether the provider needs to create a version 2 certificate."""

    @abc.abstractmethod
    def create_backend(self, module, backend):
        """Create an implementation for a backend.

        Return value must be instance of CertificateBackend.
        """