from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2SchemaModifiedCadence(_messages.Message):
    """How frequency to modify the profile when the table's schema is modified.

  Enums:
    FrequencyValueValuesEnum: Frequency to regenerate data profiles when the
      schema is modified. Defaults to monthly.
    TypesValueListEntryValuesEnum:

  Fields:
    frequency: Frequency to regenerate data profiles when the schema is
      modified. Defaults to monthly.
    types: The types of schema modifications to consider. Defaults to
      NEW_COLUMNS.
  """

    class FrequencyValueValuesEnum(_messages.Enum):
        """Frequency to regenerate data profiles when the schema is modified.
    Defaults to monthly.

    Values:
      UPDATE_FREQUENCY_UNSPECIFIED: Unspecified.
      UPDATE_FREQUENCY_NEVER: After the data profile is created, it will never
        be updated.
      UPDATE_FREQUENCY_DAILY: The data profile can be updated up to once every
        24 hours.
      UPDATE_FREQUENCY_MONTHLY: The data profile can be updated up to once
        every 30 days. Default.
    """
        UPDATE_FREQUENCY_UNSPECIFIED = 0
        UPDATE_FREQUENCY_NEVER = 1
        UPDATE_FREQUENCY_DAILY = 2
        UPDATE_FREQUENCY_MONTHLY = 3

    class TypesValueListEntryValuesEnum(_messages.Enum):
        """TypesValueListEntryValuesEnum enum type.

    Values:
      SQL_SCHEMA_MODIFICATION_UNSPECIFIED: Unused.
      NEW_COLUMNS: New columns has appeared.
      REMOVED_COLUMNS: Columns have been removed from the table.
    """
        SQL_SCHEMA_MODIFICATION_UNSPECIFIED = 0
        NEW_COLUMNS = 1
        REMOVED_COLUMNS = 2
    frequency = _messages.EnumField('FrequencyValueValuesEnum', 1)
    types = _messages.EnumField('TypesValueListEntryValuesEnum', 2, repeated=True)