from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.endpoints import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import retry
import six
def FilenameMatchesExtension(filename, extensions):
    """Checks to see if a file name matches one of the given extensions.

  Args:
    filename: The full path to the file to check
    extensions: A list of candidate extensions.

  Returns:
    True if the filename matches one of the extensions, otherwise False.
  """
    f = filename.lower()
    for ext in extensions:
        if f.endswith(ext.lower()):
            return True
    return False