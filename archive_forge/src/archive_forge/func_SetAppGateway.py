from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.beyondcorp.app import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.beyondcorp.app import util as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def SetAppGateway(unused_ref, args, request):
    """Set the app gateway to resource based string format for beta release track.

  Args:
    unused_ref: The unused request URL.
    args: arguments set by user.
    request: create connection request raised by framework.

  Returns:
    request with modified app gateway argument.
  """
    if args.calliope_command.ReleaseTrack() == base.ReleaseTrack.BETA and args.IsSpecified('app_gateway'):
        if not args.IsSpecified('project'):
            args.project = properties.VALUES.core.project.Get()
        request.googleCloudBeyondcorpAppconnectionsV1AppConnection.gateway.appGateway = APPGATEWAY_RESOURCE_NAME.format(args.project, args.location, GetVersionedConnectionReq(args, request).gateway.appGateway)
    return request