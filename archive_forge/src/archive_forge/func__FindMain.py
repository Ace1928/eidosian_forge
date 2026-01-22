from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import fnmatch
import os
import re
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app.images import config as images_config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
def _FindMain(filename):
    """Check filename for 'package main' and 'func main'.

  Args:
    filename: (str) File name to check.

  Returns:
    (bool) True if main is found in filename.
  """
    with files.FileReader(filename) as f:
        found_package = False
        found_func = False
        for line in f:
            if re.match('^package main', line):
                found_package = True
            elif re.match('^func main', line):
                found_func = True
            if found_package and found_func:
                return True
    return False