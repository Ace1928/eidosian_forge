from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import io
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import http_encoding
def CredentialFrom(message, principal):
    """Translates a dict of credential data into a message object.

  Args:
    message: The API message to use.
    principal: A string contains service account data.
  Returns:
    An ServiceAccount message object derived from credential_string.
  Raises:
    InvalidArgumentException: principal string unexpected format.
  """
    if principal == 'PROJECT_DEFAULT':
        return message.Credential(useProjectDefault=True)
    if principal.startswith('serviceAccount:'):
        service_account = message.ServiceAccount(email=principal[len('serviceAccount:'):])
        return message.Credential(serviceAccount=service_account)
    raise calliope_exceptions.InvalidArgumentException('--credential', 'credential must start with serviceAccount: or use PROJECT_DEFAULT.')