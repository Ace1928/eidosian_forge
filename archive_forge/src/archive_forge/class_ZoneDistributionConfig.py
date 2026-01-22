from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ZoneDistributionConfig(_messages.Message):
    """Zone distribution config for allocation of cluster resources.

  Enums:
    ModeValueValuesEnum: Optional. The mode of zone distribution. Defaults to
      MULTI_ZONE, when not specified.

  Fields:
    mode: Optional. The mode of zone distribution. Defaults to MULTI_ZONE,
      when not specified.
    zone: Optional. When SINGLE ZONE distribution is selected, zone field
      would be used to allocate all resources in that zone. This is not
      applicable to MULTI_ZONE, and would be ignored for MULTI_ZONE clusters.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Optional. The mode of zone distribution. Defaults to MULTI_ZONE, when
    not specified.

    Values:
      ZONE_DISTRIBUTION_MODE_UNSPECIFIED: Not Set. Default: MULTI_ZONE
      MULTI_ZONE: Distribute all resources across 3 zones picked at random,
        within the region.
      SINGLE_ZONE: Distribute all resources in a single zone. The zone field
        must be specified, when this mode is selected.
    """
        ZONE_DISTRIBUTION_MODE_UNSPECIFIED = 0
        MULTI_ZONE = 1
        SINGLE_ZONE = 2
    mode = _messages.EnumField('ModeValueValuesEnum', 1)
    zone = _messages.StringField(2)