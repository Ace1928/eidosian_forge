from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.spanner import backup_operations
from googlecloudsdk.api_lib.spanner import database_operations
from googlecloudsdk.api_lib.spanner import instance_config_operations
from googlecloudsdk.api_lib.spanner import instance_operations
from googlecloudsdk.api_lib.spanner import instance_partition_operations
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.spanner import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class AlphaCancel(Cancel):
    """Cancel a Cloud Spanner operation with ALPHA features."""
    detailed_help = {'EXAMPLES': DETAILED_HELP['EXAMPLES'] + textwrap.dedent('\n        To cancel a Cloud Spanner instance partition operation with ID auto_12345, run:\n\n          $ {command} auto_12345 --instance=my-instance-id --instance-partition=my-partition-id\n        ')}

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Please add arguments in alphabetical order except for no- or a clear-
    pair for that argument which can follow the argument itself.
    Args:
      parser: An argparse parser that you can use to add arguments that go on
        the command line after this command. Positional arguments are allowed.
    """
        super(AlphaCancel, AlphaCancel).Args(parser)
        flags.InstancePartition(positional=False, required=False, hidden=True, text='For instance partition operations, the name of the instance partition the operation is executing on.').AddToParser(parser)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        if args.instance_partition:
            return instance_partition_operations.Cancel(args.instance_partition, args.instance, args.operation)
        return super().Run(args)