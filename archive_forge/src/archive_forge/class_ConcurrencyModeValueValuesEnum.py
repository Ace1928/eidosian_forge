from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConcurrencyModeValueValuesEnum(_messages.Enum):
    """The concurrency control mode to use for this database.

    Values:
      CONCURRENCY_MODE_UNSPECIFIED: Not used.
      OPTIMISTIC: Use optimistic concurrency control by default. This mode is
        available for Cloud Firestore databases.
      PESSIMISTIC: Use pessimistic concurrency control by default. This mode
        is available for Cloud Firestore databases. This is the default
        setting for Cloud Firestore.
      OPTIMISTIC_WITH_ENTITY_GROUPS: Use optimistic concurrency control with
        entity groups by default. This is the only available mode for Cloud
        Datastore. This mode is also available for Cloud Firestore with
        Datastore Mode but is not recommended.
    """
    CONCURRENCY_MODE_UNSPECIFIED = 0
    OPTIMISTIC = 1
    PESSIMISTIC = 2
    OPTIMISTIC_WITH_ENTITY_GROUPS = 3