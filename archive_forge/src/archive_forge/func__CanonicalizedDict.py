from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.functions import secrets_config
import six
def _CanonicalizedDict(secrets_dict):
    """Canonicalizes all keys in the dict and returns a new dict.

  Args:
    secrets_dict: Existing secrets configuration dict.

  Returns:
    Canonicalized secrets configuration dict.
  """
    return collections.OrderedDict(sorted(six.iteritems({secrets_config.CanonicalizeKey(key): value for key, value in secrets_dict.items()})))