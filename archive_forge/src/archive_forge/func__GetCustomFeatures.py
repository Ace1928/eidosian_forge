from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.ssl_policies import ssl_policies_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.ssl_policies import flags
@staticmethod
def _GetCustomFeatures(args):
    """Returns the custom features specified on the command line.

    Args:
      args: The arguments passed to this command from the command line.

    Returns:
      A tuple. The first element in the tuple indicates whether custom
      features must be included in the request or not. The second element in
      the tuple specifies the list of custom features.
    """
    if args.IsSpecified('profile') and args.profile != 'CUSTOM':
        if args.IsSpecified('custom_features') and len(args.custom_features) > 0:
            raise exceptions.InvalidArgumentException('--custom-features', 'Custom features cannot be specified when using non-CUSTOM profiles.')
        return (True, [])
    elif args.IsSpecified('custom_features'):
        return (True, args.custom_features)
    else:
        return (False, [])