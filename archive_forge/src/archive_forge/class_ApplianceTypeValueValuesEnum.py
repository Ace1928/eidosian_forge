from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplianceTypeValueValuesEnum(_messages.Enum):
    """Required. The appliance type of identity source. Can be vCenter or
    NSX-T.

    Values:
      APPLIANCE_TYPE_UNSPECIFIED: The default value. This value should never
        be used.
      VCENTER: A vCenter appliance.
    """
    APPLIANCE_TYPE_UNSPECIFIED = 0
    VCENTER = 1