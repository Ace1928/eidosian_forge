from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionValueValuesEnum(_messages.Enum):
    """Indicates the user-supplied encryption option of this VLAN attachment
    (interconnectAttachment). Can only be specified at attachment creation for
    PARTNER or DEDICATED attachments. Possible values are: - NONE - This is
    the default value, which means that the VLAN attachment carries
    unencrypted traffic. VMs are able to send traffic to, or receive traffic
    from, such a VLAN attachment. - IPSEC - The VLAN attachment carries only
    encrypted traffic that is encrypted by an IPsec device, such as an HA VPN
    gateway or third-party IPsec VPN. VMs cannot directly send traffic to, or
    receive traffic from, such a VLAN attachment. To use *HA VPN over Cloud
    Interconnect*, the VLAN attachment must be created with this option.

    Values:
      IPSEC: The interconnect attachment will carry only encrypted traffic
        that is encrypted by an IPsec device such as HA VPN gateway; VMs
        cannot directly send traffic to or receive traffic from such an
        interconnect attachment. To use HA VPN over Cloud Interconnect, the
        interconnect attachment must be created with this option.
      NONE: This is the default value, which means the Interconnect Attachment
        will carry unencrypted traffic. VMs will be able to send traffic to or
        receive traffic from such interconnect attachment.
    """
    IPSEC = 0
    NONE = 1