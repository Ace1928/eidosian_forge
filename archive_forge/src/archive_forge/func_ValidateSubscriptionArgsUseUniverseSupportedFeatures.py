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
def ValidateSubscriptionArgsUseUniverseSupportedFeatures(args):
    """Raises an exception if args has unsupported features in non default universe.

  Args:
    args (argparse.Namespace): Parsed arguments

  Raises:
    InvalidArgumentException: if invalid flags are set in current universe.
  """
    if properties.IsDefaultUniverse():
        return
    universe_domain = properties.GetUniverseDomain()
    _RaiseExceptionIfContains(args, universe_domain, NON_GDU_DISABLED_SUBSCRIPION_FLAG_FEATURE_MAP)