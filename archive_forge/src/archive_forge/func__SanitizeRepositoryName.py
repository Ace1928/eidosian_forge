from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import re
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.core import log
def _SanitizeRepositoryName(name):
    """Sanitizes the given name to make it valid as an image repository.

  As explained in
  https://docs.docker.com/engine/reference/commandline/tag/#extended-description,
  Valid name may contain only lowercase letters, digits and separators.
  A separator is defined as a period, one or two underscores, or one or more
  dashes. A name component may not start or end with a separator.

  This method will replace the illegal characters in the given name and strip
  starting and ending separator characters.

  Args:
    name: str, the name to sanitize.

  Returns:
    A sanitized name.
  """
    return re.sub('[._][._]+|[^a-z0-9._-]+', '.', name.lower()).strip('._-')