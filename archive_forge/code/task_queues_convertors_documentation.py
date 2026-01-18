from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.tasks import constants
import six
Converts task age limit values to the format CT expects.

  Args:
    value: A string value representing the task age limit. For example, '2.5m',
      '1h', '8s', etc.

  Returns:
    The string representing the time to the nearest second with 's' appended
    at the end.
  