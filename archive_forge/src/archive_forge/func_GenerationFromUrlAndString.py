from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import stat
import sys
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.utils import system_util
from gslib.utils import text_util
def GenerationFromUrlAndString(url, generation):
    """Decodes a generation from a StorageURL and a generation string.

  This is used to represent gs and s3 versioning.

  Args:
    url: StorageUrl representing the object.
    generation: Long or string representing the object's generation or
                version.

  Returns:
    Valid generation string for use in URLs.
  """
    if url.scheme == 's3' and generation:
        return text_util.DecodeLongAsString(generation)
    return generation