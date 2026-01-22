from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.functions import secrets_config
import six
def _ParseSecretRef(secret_ref):
    """Splits a secret version resource into its components.

  Args:
    secret_ref: Secret version resource reference.

  Returns:
    A dict with entries for project, secret and version.
  """
    return _SECRET_VERSION_RESOURCE_PATTERN.search(secret_ref).groupdict()