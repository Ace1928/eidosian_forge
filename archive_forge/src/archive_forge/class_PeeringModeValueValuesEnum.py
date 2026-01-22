from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PeeringModeValueValuesEnum(_messages.Enum):
    """Optional. The network connect mode of the ManagementServer instance.
    For this version, only PRIVATE_SERVICE_ACCESS is supported.

    Values:
      PEERING_MODE_UNSPECIFIED: Peering mode not set.
      PRIVATE_SERVICE_ACCESS: Connect using Private Service Access to the
        Management Server. Private services access provides an IP address
        range for multiple Google Cloud services, including Cloud BackupDR.
    """
    PEERING_MODE_UNSPECIFIED = 0
    PRIVATE_SERVICE_ACCESS = 1