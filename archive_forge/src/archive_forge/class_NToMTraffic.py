from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NToMTraffic(_messages.Message):
    """Predefined traffic shape in which each `src_group` member sends traffic
  to each `dst_group` member. For example: * 1 entry in `src` and many entries
  in `dst` represent `One to Many` traffic. * 1 entry in `dst` and many
  entries in `src` represent `Many to One` traffic.

  Enums:
    TrafficDirectionValueValuesEnum: If UNSPECIFIED or UNIDIRECTIONAL, each
      `src_group` member sends traffic to each `dst_group` member. If
      BIDIRECTIONAL, each `dst_group` member additionally sends traffic to
      each `src_group` member.

  Fields:
    dstGroup: `dst` coordinates receiving data from all coordinates in
      `src_group`.
    srcGroup: `src` coordinates sending data to all coordinates in
      `dst_group`.
    trafficDirection: If UNSPECIFIED or UNIDIRECTIONAL, each `src_group`
      member sends traffic to each `dst_group` member. If BIDIRECTIONAL, each
      `dst_group` member additionally sends traffic to each `src_group`
      member.
  """

    class TrafficDirectionValueValuesEnum(_messages.Enum):
        """If UNSPECIFIED or UNIDIRECTIONAL, each `src_group` member sends
    traffic to each `dst_group` member. If BIDIRECTIONAL, each `dst_group`
    member additionally sends traffic to each `src_group` member.

    Values:
      TRAFFIC_DIRECTION_UNSPECIFIED: Traffic direction is not specified
      TRAFFIC_DIRECTION_UNIDIRECTIONAL: Traffic is sent in one direction.
      TRAFFIC_DIRECTION_BIDIRECTIONAL: Traffic is sent in both directions.
    """
        TRAFFIC_DIRECTION_UNSPECIFIED = 0
        TRAFFIC_DIRECTION_UNIDIRECTIONAL = 1
        TRAFFIC_DIRECTION_BIDIRECTIONAL = 2
    dstGroup = _messages.MessageField('CoordinateList', 1)
    srcGroup = _messages.MessageField('CoordinateList', 2)
    trafficDirection = _messages.EnumField('TrafficDirectionValueValuesEnum', 3)