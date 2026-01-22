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
class _OnDiskCertConfigImpl(_ConfigImpl):
    """Represents the configurations associated with context aware access through a certificate on disk.

  Both the encrypted and unencrypted certs need to be generated to support HTTP
  API clients and gRPC API clients, respectively.

  Only one instance of Config can be created for the program.
  """

    def __init__(self, config_path, client_cert_bytes, client_key_bytes, encrypted_client_cert_path, encrypted_client_cert_password):
        super(_OnDiskCertConfigImpl, self).__init__(ConfigType.ON_DISK_CERTIFICATE)
        self.config_path = config_path
        self.client_cert_bytes = client_cert_bytes
        self.client_key_bytes = client_key_bytes
        self.encrypted_client_cert_path = encrypted_client_cert_path
        self.encrypted_client_cert_password = encrypted_client_cert_password
        atexit.register(self.CleanUp)

    def CleanUp(self):
        """Cleanup any files or resource provisioned during config init."""
        if self.encrypted_client_cert_path is not None and os.path.exists(self.encrypted_client_cert_path):
            try:
                os.remove(self.encrypted_client_cert_path)
                log.debug('unprovisioned client cert - %s', self.encrypted_client_cert_path)
            except files.Error as e:
                log.error('failed to remove client certificate - %s', e)