from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1RedirectWritesStepDetails(_messages.Message):
    """Details for the `REDIRECT_WRITES` step.

  Enums:
    ConcurrencyModeValueValuesEnum: Ths concurrency mode for this database.

  Fields:
    concurrencyMode: Ths concurrency mode for this database.
  """

    class ConcurrencyModeValueValuesEnum(_messages.Enum):
        """Ths concurrency mode for this database.

    Values:
      CONCURRENCY_MODE_UNSPECIFIED: Unspecified.
      PESSIMISTIC: Pessimistic concurrency.
      OPTIMISTIC: Optimistic concurrency.
      OPTIMISTIC_WITH_ENTITY_GROUPS: Optimistic concurrency with entity
        groups.
    """
        CONCURRENCY_MODE_UNSPECIFIED = 0
        PESSIMISTIC = 1
        OPTIMISTIC = 2
        OPTIMISTIC_WITH_ENTITY_GROUPS = 3
    concurrencyMode = _messages.EnumField('ConcurrencyModeValueValuesEnum', 1)