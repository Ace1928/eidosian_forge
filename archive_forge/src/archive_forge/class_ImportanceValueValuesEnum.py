from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportanceValueValuesEnum(_messages.Enum):
    """DO NOT USE. This is an experimental field.

    Values:
      LOW: Allows data caching, batching, and aggregation. It provides higher
        performance with higher data loss risk.
      HIGH: Disables data aggregation to minimize data loss. It is for
        operations that contains significant monetary value or audit trail.
        This feature only applies to the client libraries.
      DEBUG: Deprecated. Do not use. Disables data aggregation and enables
        additional validation logic. It should only be used during the
        onboarding process. It is only available to Google internal services,
        and the service must be approved by chemist-dev@google.com in order to
        use this level.
      PROMOTED: Used internally by Chemist.
    """
    LOW = 0
    HIGH = 1
    DEBUG = 2
    PROMOTED = 3