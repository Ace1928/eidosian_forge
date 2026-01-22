from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.parser_errors import DetailedArgumentError
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
def _ParseAssignInboundPublicIp(assign_inbound_public_ip):
    """Parses the assign_inbound_public_ip flag.

  Args:
    assign_inbound_public_ip: string, the Public-IP mode to use.

  Returns:
    boolean, whether or not Public-IP is enabled.

  Raises:
    ValueError if try to use any other value besides NO_PUBLIC_IP during
    instance creation, or if use an unrecognized argument.
  """
    if assign_inbound_public_ip == 'NO_PUBLIC_IP':
        return False
    if assign_inbound_public_ip == 'ASSIGN_IPV4':
        return True
    raise DetailedArgumentError('Unrecognized argument. Please use NO_PUBLIC_IP or ASSIGN_IPV4.')