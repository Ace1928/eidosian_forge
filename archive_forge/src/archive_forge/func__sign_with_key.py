from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import hashlib
import json
import urllib.parse
from googlecloudsdk.api_lib.util import apis_internal
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core import requests as core_requests
from googlecloudsdk.core import transport
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import transports
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import requests
def _sign_with_key(key, string_to_sign):
    """Generates a signature using OpenSSL.crypto.

  Args:
    key (crypto.PKey): Key for the signing service account.
    string_to_sign (str): String to sign.

  Returns:
      A raw signature for the specified string.
  """
    from OpenSSL import crypto
    return crypto.sign(key, string_to_sign.encode('utf-8'), _DIGEST)