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
def SetConnectors(unused_ref, args, request):
    """Set the connectors to resource based string format.

  Args:
    unused_ref: The unused request URL.
    args: arguments set by user.
    request: create connection request raised by framework.

  Returns:
    request with modified connectors argument.
  """
    if args.IsSpecified('connectors'):
        if not args.IsSpecified('project'):
            args.project = properties.VALUES.core.project.Get()
        for index, connector in enumerate(GetVersionedConnectionReq(args, request).connectors):
            if args.calliope_command.ReleaseTrack() == base.ReleaseTrack.ALPHA:
                request.googleCloudBeyondcorpAppconnectionsV1alphaAppConnection.connectors[index] = APPCONNECTOR_RESOURCE_NAME.format(args.project, args.location, connector)
            else:
                request.googleCloudBeyondcorpAppconnectionsV1AppConnection.connectors[index] = APPCONNECTOR_RESOURCE_NAME.format(args.project, args.location, connector)
    return request