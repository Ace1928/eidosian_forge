from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.identity import cloudidentity_client as ci_client
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.identity.groups import hooks as groups_hooks
from googlecloudsdk.core.util import times
def FormatDateTime(duration):
    """Return RFC3339 string for datetime that is now + given duration.

  Args:
    duration: string ISO 8601 duration, e.g. 'P5D' for period 5 days.

  Returns:
    string timestamp

  """
    fmt = '%Y-%m-%dT%H:%M:%S.%3f%Oz'
    return times.FormatDateTime(times.ParseDateTime(duration, tzinfo=times.UTC), fmt=fmt)