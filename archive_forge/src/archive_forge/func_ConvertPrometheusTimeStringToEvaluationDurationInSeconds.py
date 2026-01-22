from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ConvertPrometheusTimeStringToEvaluationDurationInSeconds(time_string):
    """Converts Prometheus time to duration JSON string.

  Args:
    time_string: String provided by the alert rule YAML file defining time
      (ex:1h30m)

  Raises:
    ValueError: If the provided time_string is not a multiple of 30 seconds or
    is less than 30 seconds.

  Returns:
    Duration proto string representing the adjusted seconds (multiple of 30
    seconds) value of the provided time_string
  """
    seconds = ConvertIntervalToSeconds(time_string)
    if seconds < 30:
        raise ValueError('{time_string} converted to {seconds}s is less than 30 seconds.'.format(time_string=time_string, seconds=seconds))
    elif seconds % 30 != 0:
        raise ValueError('{} converted to {}s is not a multiple of 30 seconds.'.format(time_string, seconds))
    return _FormatDuration(seconds)