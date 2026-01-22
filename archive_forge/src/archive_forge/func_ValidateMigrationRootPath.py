from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateMigrationRootPath(root_path):
    """Validates the root path inside the Cloud Storage bucket used for CDC during migration, must start with a forward slash ('/') character.

  Args:
    root_path: the root path inside the Cloud Storage bucket.

  Returns:
    the root path.
  Raises:
    BadArgumentException: when the root path is invalid.
  """
    pattern = re.compile('^/([^\\n\\r]*)$')
    if not pattern.match(root_path):
        raise exceptions.BadArgumentException('--root-path', 'Invalid root path')
    return root_path