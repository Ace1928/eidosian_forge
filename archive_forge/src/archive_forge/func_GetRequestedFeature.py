from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.calliope import actions as calliope_actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetRequestedFeature(messages, feature_arg):
    """Converts interconnect feature type flag to a message enum.

  Args:
    messages: The API messages holder.
    feature_arg: The feature type flag value.

  Returns:
    A RequestedFeaturesValueListEntryValuesEnum of the flag value.
  """
    if feature_arg == 'MACSEC':
        return messages.Interconnect.RequestedFeaturesValueListEntryValuesEnum('IF_MACSEC')
    return None