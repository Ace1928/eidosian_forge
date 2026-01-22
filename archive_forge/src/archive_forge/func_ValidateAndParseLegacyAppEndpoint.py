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
def ValidateAndParseLegacyAppEndpoint(unused_ref, args, request):
    """Validates app endpoint format and sets endpoint host and port after parsing.

  Args:
    unused_ref: The unused request URL.
    args: arguments set by user.
    request: create connection request raised by framework.

  Returns:
    request with modified application endpoint host and port argument.

  Raises:
    ApplicationEndpointParseError:
  """
    if args.IsSpecified('application_endpoint'):
        endpoint_array = args.application_endpoint.split(':')
        if len(endpoint_array) == 2 and endpoint_array[1].isdigit():
            messages = api_util.GetMessagesModule(args.calliope_command.ReleaseTrack())
            if request.connection is None:
                request.connection = messages.Connection()
            if request.connection.applicationEndpoint is None:
                request.connection.applicationEndpoint = messages.ApplicationEndpoint()
            request.connection.applicationEndpoint.host = endpoint_array[0]
            request.connection.applicationEndpoint.port = int(endpoint_array[1])
        else:
            raise ApplicationEndpointParseError(APP_ENDPOINT_PARSE_ERROR.format(args.application_endpoint))
    return request