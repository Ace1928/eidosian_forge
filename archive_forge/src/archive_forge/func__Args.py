from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _Args(parser):
    """Register flags for this command.

  Args:
    parser: an argparse.ArgumentParser-like object. It is mocked out in order to
      capture some information, but behaves like an ArgumentParser.
  """
    flags.AddAsyncFlag(parser)
    flags.AddNodePoolNameArg(parser, 'The name of the node pool to rollback.')
    flags.AddNodePoolClusterFlag(parser, 'The cluster from which to rollback the node pool.')
    flags.AddRespectPodDisruptionBudgetFlag(parser)
    parser.add_argument('--timeout', type=int, default=1800, hidden=True, help='THIS ARGUMENT NEEDS HELP TEXT.')