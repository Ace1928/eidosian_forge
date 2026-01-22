from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.certificate_manager import api_client
from googlecloudsdk.core.util import times
def ParseIso8601LifetimeFlag(value):
    """Parses the ISO 8601 lifetime argument.

  Args:
    value: An ISO 8601 valid value.

  Returns:
    modified value as expected by the API
  """
    return times.FormatDurationForJson(times.ParseDuration(value))