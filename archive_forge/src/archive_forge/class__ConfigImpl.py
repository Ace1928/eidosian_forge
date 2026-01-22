from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import enum
import json
import os
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport import _mtls_helper
from googlecloudsdk.command_lib.auth import enterprise_certificate_config
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
class _ConfigImpl(object):
    """Represents the configurations associated with context aware access.

  Both the encrypted and unencrypted certs need to be generated to support HTTP
  API clients and gRPC API clients, respectively.

  Only one instance of Config can be created for the program.
  """

    @classmethod
    def Load(cls):
        """Loads the context aware config."""
        if not properties.VALUES.context_aware.use_client_certificate.GetBool():
            return None
        certificate_config_file_path = _GetCertificateConfigFile()
        if certificate_config_file_path:
            log.debug('enterprise certificate is used for mTLS')
            return _EnterpriseCertConfigImpl(certificate_config_file_path)
        log.debug('on disk certificate is used for mTLS')
        config_path = _AutoDiscoveryFilePath()
        cert_bytes, key_bytes = SSLCredentials(config_path)
        encrypted_cert_path, password = EncryptedSSLCredentials(config_path)
        return _OnDiskCertConfigImpl(config_path, cert_bytes, key_bytes, encrypted_cert_path, password)

    def __init__(self, config_type):
        self.config_type = config_type