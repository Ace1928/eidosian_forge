from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def GetProfileFlag(default=None):
    """Returns the flag for specifying the SSL policy profile."""
    return base.Argument('--profile', choices={'COMPATIBLE': 'Compatible profile. Allows the broadest set of clients, even those which support only out-of-date SSL features, to negotiate SSL with the load balancer.', 'MODERN': 'Modern profile. Supports a wide set of SSL features, allowing modern clients to negotiate SSL.', 'RESTRICTED': 'Restricted profile. Supports a reduced set of SSL features, intended to meet stricter compliance requirements.', 'CUSTOM': 'Custom profile. Allows customization by selecting only the features which are required. The list of all available features can be obtained using:\n\n  gcloud compute ssl-policies list-available-features\n'}, default=default, help='SSL policy profile. Changing profile from CUSTOM to COMPATIBLE|MODERN|RESTRICTED will clear the custom-features field.')