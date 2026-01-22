from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
def AddChangeStreamConfigUpdateTableArgs():
    """Adds the change stream commands to update table CLI.

  This can't be defined in the yaml because that automatically generates the
  inverse for any boolean args and we don't want the nonsensical
  'no-clear-change-stream-retention-period`. We use store_const to only allow
  `clear-change-stream-retention-period` or `change-stream-retention-period`
  arguments

  Returns:
    Argument group containing change stream args
  """
    argument_group = base.ArgumentGroup(mutex=True)
    argument_group.AddArgument(base.Argument('--clear-change-stream-retention-period', help='This disables the change stream and eventually removes the change stream data.', action='store_const', const=True))
    argument_group.AddArgument(base.Argument('--change-stream-retention-period', help='The length of time to retain change stream data for the table, in the range of [1 day, 7 days]. Acceptable units are days (d), hours (h), minutes (m), and seconds (s). If not already specified, enables a change stream for the table. Examples: `5d` or `48h`.'))
    return [argument_group]