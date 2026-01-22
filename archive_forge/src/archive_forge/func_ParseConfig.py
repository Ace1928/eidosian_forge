from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def ParseConfig(unused_ref, args, request):
    """Parse client connector service config."""
    if args.IsSpecified('config_from_file'):
        return GetConfigFromFile(args, request)
    elif args.IsSpecified('ingress_config') and args.IsSpecified('egress_peered_vpc'):
        return ConstructRequest(json.loads(args.ingress_config), json.loads(args.egress_peered_vpc), args.display_name, args, request)
    else:
        raise exceptions.Error('Incorrect arguments provided. Try --help.')