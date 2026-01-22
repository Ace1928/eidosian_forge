from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.infra_manager import errors
from googlecloudsdk.core.util import files
import hcl2
import six
def ParseTFvarFile(filename):
    """Parses a `tfvar` file and returns a dictionary of configuration values.

  Args:
    filename: The path to the `tfvar` file.

  Returns:
    A dictionary of configuration values.
  """
    try:
        f = files.ReadFileContents(filename)
        config = hcl2.loads(f)
        return config
    except Exception as e:
        raise errors.InvalidDataError('Error encountered while parsing ' + filename + ': ' + six.text_type(e))