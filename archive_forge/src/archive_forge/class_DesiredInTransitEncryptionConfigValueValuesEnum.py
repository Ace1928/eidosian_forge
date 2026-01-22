from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DesiredInTransitEncryptionConfigValueValuesEnum(_messages.Enum):
    """Specify the details of in-transit encryption.

    Values:
      IN_TRANSIT_ENCRYPTION_CONFIG_UNSPECIFIED: Unspecified, will be inferred
        as default - IN_TRANSIT_ENCRYPTION_UNSPECIFIED.
      IN_TRANSIT_ENCRYPTION_DISABLED: In-transit encryption is disabled.
      IN_TRANSIT_ENCRYPTION_INTER_NODE_TRANSPARENT: Data in-transit is
        encrypted using inter-node transparent encryption.
    """
    IN_TRANSIT_ENCRYPTION_CONFIG_UNSPECIFIED = 0
    IN_TRANSIT_ENCRYPTION_DISABLED = 1
    IN_TRANSIT_ENCRYPTION_INTER_NODE_TRANSPARENT = 2