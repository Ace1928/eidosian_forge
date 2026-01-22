from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def ValidateFilterString(args):
    """Raises an exception if filter string is empty.

  Args:
    args (argparse.Namespace): Parsed arguments

  Raises:
    InvalidArgumentException: if filter string is empty.
  """
    if args.message_filter is not None and (not args.message_filter):
        raise exceptions.InvalidArgumentException('--message-filter', 'Filter string must be non-empty. If you do not want a filter, ' + 'do not set the --message-filter argument.')