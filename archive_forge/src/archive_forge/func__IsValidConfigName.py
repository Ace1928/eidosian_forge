from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import re
import threading
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import properties_file
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
def _IsValidConfigName(config_name, allow_reserved):
    """Determines if the given configuration name conforms to the standard.

  Args:
    config_name: str, The name to check.
    allow_reserved: bool, Allows the given name to be one of the reserved
      configuration names.

  Returns:
    True if valid, False otherwise.
  """
    if config_name in _RESERVED_CONFIG_NAMES:
        if not allow_reserved:
            return False
    elif not re.match(_VALID_CONFIG_NAME_REGEX, config_name):
        return False
    return True