from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeTypeValueValuesEnum(_messages.Enum):
    """Required. Runtime type of the Apigee organization based on the Apigee
    subscription purchased.

    Values:
      RUNTIME_TYPE_UNSPECIFIED: Runtime type not specified.
      CLOUD: Google-managed Apigee runtime.
      HYBRID: User-managed Apigee hybrid runtime.
    """
    RUNTIME_TYPE_UNSPECIFIED = 0
    CLOUD = 1
    HYBRID = 2