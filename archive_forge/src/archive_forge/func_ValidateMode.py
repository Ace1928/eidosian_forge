from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six.moves.urllib.parse
def ValidateMode(mode):
    """Validates the mode input.

  Args:
    mode: String indicating the rendering mode of the instance. Allowed values
      are 3d and ar.

  Returns:
    True if the mode is supported by ISXR, False otherwise.
  """
    mode = mode.lower()
    if mode == '3d' or mode == 'ar':
        return True
    raise exceptions.InvalidArgumentException('--mode', 'mode must be 3d or ar')