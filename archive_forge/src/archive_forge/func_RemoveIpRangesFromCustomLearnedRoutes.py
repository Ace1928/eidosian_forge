from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def RemoveIpRangesFromCustomLearnedRoutes(messages, peer, ip_ranges):
    """Removes all specified IP address ranges from a peer's custom learned routes.

  Raises an error if any of the specified custom learned route IP address ranges
  were not found in the peer's IP ranges set. The IP address range search is
  done by exact text match.

  Args:
    messages: API messages holder.
    peer: the peer being modified.
    ip_ranges: the custom learned route IP address ranges to remove.

  Raises:
    IpRangeNotFoundError: if any IP address range was not found in the peer.
  """
    for ip_range in ip_ranges:
        if ip_range not in [r.range for r in peer.customLearnedIpRanges]:
            raise IpRangeNotFoundError(messages, messages.RouterBgpPeer, _CUSTOM_LEARNED_ROUTE_IP_RANGE_NOT_FOUND_ERROR_MESSAGE, ip_range)
    peer.customLearnedIpRanges = [r for r in peer.customLearnedIpRanges if r.range not in ip_ranges]