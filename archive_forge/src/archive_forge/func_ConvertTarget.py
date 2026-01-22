from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.tasks import constants
import six
def ConvertTarget(value):
    """Converts target to that format that Cloud Tasks APIs expect.

  Args:
    value: A string representing the service or version_dot_service.

  Returns:
    An ordered dict with parsed values for service and target if it exists.

  Raises:
    ValueError: If the input provided for target is not in the format expected.
  """
    targets = value.split('.')
    if len(targets) == 1:
        return collections.OrderedDict({'service': targets[0]})
    elif len(targets) == 2:
        return collections.OrderedDict({'service': targets[1], 'version': targets[0]})
    raise ValueError('Unsupported value received for target {}'.format(value))