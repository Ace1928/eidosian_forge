from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import instances
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class SwitchoverBeta(base.Command):
    """Switches over a Cloud SQL instance to one of its replicas.

  Switches over a Cloud SQL instance to one of its replicas. Only supported on
  Cloud SQL for SQL Server and MySQL instances.
  """
    detailed_help = DETAILED_HELP_BETA

    def Run(self, args):
        return RunBaseSwitchoverCommand(args)

    @staticmethod
    def Args(parser):
        """Args is called by calliope to gather arguments for this command.

    Args:
      parser: An argparse parser that you can use to add arguments that go
          on the command line after this command. Positional arguments are
          allowed.
    """
        AddBaseArgs(parser)