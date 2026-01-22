from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def ParseUpdatedConfig(unused_ref, args, request):
    """Parse client connector service config for update request."""
    if args.IsSpecified('config_from_file'):
        request = command_util.AddFieldToUpdateMask('ingress.config.destination_routes', request)
        request = command_util.AddFieldToUpdateMask('display_name', request)
        return GetConfigFromFile(args, request)
    elif args.IsSpecified('ingress_config') or args.IsSpecified('display_name'):
        if args.IsSpecified('ingress_config'):
            request = command_util.AddFieldToUpdateMask('ingress.config.destination_routes', request)
        if args.IsSpecified('display_name'):
            request = command_util.AddFieldToUpdateMask('display_name', request)
        ingress_config = json.loads(args.ingress_config) if args.IsSpecified('ingress_config') else None
        return ConstructRequest(ingress_config, None, args.display_name, args, request)
    else:
        raise exceptions.Error('Incorrect arguments provided. Try --help.')