from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import re
import textwrap
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def Base64EncodedFileContents(filename):
    """Reads the provided file, and returns its contents, base64-encoded.

  Args:
    filename: The path to the file, absolute or relative to the current working
      directory.

  Returns:
    A string, the contents of filename, base64-encoded.

  Raises:
   files.Error: if the file cannot be read.
  """
    return base64.b64encode(files.ReadBinaryFileContents(files.ExpandHomeDir(filename)))