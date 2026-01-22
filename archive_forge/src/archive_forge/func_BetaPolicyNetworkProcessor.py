from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.dns import flags
import ipaddr
def BetaPolicyNetworkProcessor(parsed_value):
    """Build Beta PolicyNetwork message from parsed_value."""
    return PolicyNetworkProcessor(parsed_value, version='v1beta2')