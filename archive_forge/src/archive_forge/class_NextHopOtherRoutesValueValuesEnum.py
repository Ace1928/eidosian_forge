from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NextHopOtherRoutesValueValuesEnum(_messages.Enum):
    """Optional. Other routes that will be referenced to determine the next
    hop of the packet.

    Values:
      OTHER_ROUTES_UNSPECIFIED: Default value.
      DEFAULT_ROUTING: Use the routes from the default routing tables (system-
        generated routes, custom routes, peering route) to determine the next
        hop. This will effectively exclude matching packets being applied on
        other PBRs with a lower priority.
    """
    OTHER_ROUTES_UNSPECIFIED = 0
    DEFAULT_ROUTING = 1