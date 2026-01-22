from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReadReplicasModeValueValuesEnum(_messages.Enum):
    """Optional. Read replicas mode for the instance. Defaults to
    READ_REPLICAS_DISABLED.

    Values:
      READ_REPLICAS_MODE_UNSPECIFIED: If not set, Memorystore Redis backend
        will default to READ_REPLICAS_DISABLED.
      READ_REPLICAS_DISABLED: If disabled, read endpoint will not be provided
        and the instance cannot scale up or down the number of replicas.
      READ_REPLICAS_ENABLED: If enabled, read endpoint will be provided and
        the instance can scale up and down the number of replicas. Not valid
        for basic tier.
    """
    READ_REPLICAS_MODE_UNSPECIFIED = 0
    READ_REPLICAS_DISABLED = 1
    READ_REPLICAS_ENABLED = 2