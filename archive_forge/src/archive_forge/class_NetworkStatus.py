from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkStatus(_messages.Message):
    """NetworkStatus has a list of status for the subnets under the current
  network.

  Enums:
    MacsecStatusInternalLinksValueValuesEnum: The MACsec status of internal
      links.

  Fields:
    macsecStatusInternalLinks: The MACsec status of internal links.
    subnetStatus: A list of status for the subnets under the current network.
  """

    class MacsecStatusInternalLinksValueValuesEnum(_messages.Enum):
        """The MACsec status of internal links.

    Values:
      MACSEC_STATUS_UNSPECIFIED: MACsec status not specified, likely due to
        missing metrics.
      SECURE: All relevant links have at least one MACsec session up.
      UNSECURE: At least one relevant link does not have any MACsec sessions
        up.
    """
        MACSEC_STATUS_UNSPECIFIED = 0
        SECURE = 1
        UNSECURE = 2
    macsecStatusInternalLinks = _messages.EnumField('MacsecStatusInternalLinksValueValuesEnum', 1)
    subnetStatus = _messages.MessageField('SubnetStatus', 2, repeated=True)