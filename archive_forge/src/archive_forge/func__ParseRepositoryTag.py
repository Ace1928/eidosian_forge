from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import re
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.core import log
def _ParseRepositoryTag(image_name):
    """Parses out the repository and tag from a Docker image name.

  Args:
    image_name: (str) The full name of an image, expected to be in a format of
      "repository[:tag]"

  Returns:
    A (repository, tag) tuple representing the parsed result.
    None repository means the image name is invalid; tag may be None if it isn't
    present in the given image name.
  """
    if image_name.count(':') > 2:
        return (None, None)
    parts = image_name.rsplit(':', 1)
    if len(parts) == 2 and '/' not in parts[1]:
        return tuple(parts)
    return (image_name, None)