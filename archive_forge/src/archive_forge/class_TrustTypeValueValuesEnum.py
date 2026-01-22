from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrustTypeValueValuesEnum(_messages.Enum):
    """Required. The type of trust represented by the trust resource.

    Values:
      TRUST_TYPE_UNSPECIFIED: Not set.
      FOREST: The forest trust.
      EXTERNAL: The external domain trust.
    """
    TRUST_TYPE_UNSPECIFIED = 0
    FOREST = 1
    EXTERNAL = 2