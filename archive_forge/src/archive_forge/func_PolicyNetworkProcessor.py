from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def PolicyNetworkProcessor(parsed_value, version='v1'):
    """Build PolicyNetwork message from parsed_value."""
    messages = GetMessages(version)
    if not parsed_value:
        return []
    return [messages.PolicyNetwork(networkUrl=network_ref.SelfLink()) for network_ref in parsed_value]