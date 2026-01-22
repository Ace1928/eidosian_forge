from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RouteStatusValueValuesEnum(_messages.Enum):
    """[Output only] The status of the route.

    Values:
      ACTIVE: This route is processed and active.
      DROPPED: The route is dropped due to the VPC exceeding the dynamic route
        limit. For dynamic route limit, please refer to the Learned route
        example
      INACTIVE: This route is processed but inactive due to failure from the
        backend. The backend may have rejected the route
      PENDING: This route is being processed internally. The status will
        change once processed.
    """
    ACTIVE = 0
    DROPPED = 1
    INACTIVE = 2
    PENDING = 3