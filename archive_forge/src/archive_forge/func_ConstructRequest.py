from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
def ConstructRequest(ingress_config, egress_vpc, display_name, args, request):
    """Construct request from the given client connector service config."""
    messages = api_util.GetMessagesModule(args.calliope_command.ReleaseTrack())
    if request.clientConnectorService is None:
        request.clientConnectorService = messages.ClientConnectorService()
    if request.clientConnectorService.ingress is None:
        request.clientConnectorService.ingress = messages.Ingress()
    if request.clientConnectorService.ingress.config is None and ingress_config is not None:
        request.clientConnectorService.ingress.config = messages_util.DictToMessageWithErrorCheck(ingress_config, messages.Config)
    if request.clientConnectorService.egress is None:
        request.clientConnectorService.egress = messages.Egress()
    if request.clientConnectorService.egress.peeredVpc is None and egress_vpc is not None:
        request.clientConnectorService.egress.peeredVpc = messages_util.DictToMessageWithErrorCheck(egress_vpc, messages.PeeredVpc)
    if args.IsSpecified('display_name') or display_name is not None:
        request.clientConnectorService.displayName = display_name
    return request