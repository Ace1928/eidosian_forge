from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import subprocess
import sys
from googlecloudsdk.core.util import files
def ModuleToPath(module_name):
    """Converts the supplied python module into corresponding python file.

  Args:
    module_name: (str) A python module name (separated by dots)

  Returns:
    A string representing a python file path.
  """
    return module_name.replace('.', os.path.sep) + '.py'