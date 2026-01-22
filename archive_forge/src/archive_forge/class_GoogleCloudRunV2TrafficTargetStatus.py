from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2TrafficTargetStatus(_messages.Message):
    """Represents the observed state of a single `TrafficTarget` entry.

  Enums:
    TypeValueValuesEnum: The allocation type for this traffic target.

  Fields:
    percent: Specifies percent of the traffic to this Revision.
    revision: Revision to which this traffic is sent.
    tag: Indicates the string used in the URI to exclusively reference this
      target.
    type: The allocation type for this traffic target.
    uri: Displays the target URI.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The allocation type for this traffic target.

    Values:
      TRAFFIC_TARGET_ALLOCATION_TYPE_UNSPECIFIED: Unspecified instance
        allocation type.
      TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST: Allocates instances to the
        Service's latest ready Revision.
      TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION: Allocates instances to a
        Revision by name.
    """
        TRAFFIC_TARGET_ALLOCATION_TYPE_UNSPECIFIED = 0
        TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST = 1
        TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION = 2
    percent = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    revision = _messages.StringField(2)
    tag = _messages.StringField(3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)
    uri = _messages.StringField(5)