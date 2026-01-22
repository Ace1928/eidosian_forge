from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ReconcileTagsMetadata(_messages.Message):
    """Long-running operation metadata message returned by the ReconcileTags.

  Enums:
    StateValueValuesEnum: State of the reconciliation operation.

  Messages:
    ErrorsValue: Maps the name of each tagged column (or empty string for a
      sole entry) to tagging operation status.

  Fields:
    errors: Maps the name of each tagged column (or empty string for a sole
      entry) to tagging operation status.
    state: State of the reconciliation operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the reconciliation operation.

    Values:
      RECONCILIATION_STATE_UNSPECIFIED: Default value. This value is unused.
      RECONCILIATION_QUEUED: The reconciliation has been queued and awaits for
        execution.
      RECONCILIATION_IN_PROGRESS: The reconciliation is in progress.
      RECONCILIATION_DONE: The reconciliation has been finished.
    """
        RECONCILIATION_STATE_UNSPECIFIED = 0
        RECONCILIATION_QUEUED = 1
        RECONCILIATION_IN_PROGRESS = 2
        RECONCILIATION_DONE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ErrorsValue(_messages.Message):
        """Maps the name of each tagged column (or empty string for a sole entry)
    to tagging operation status.

    Messages:
      AdditionalProperty: An additional property for a ErrorsValue object.

    Fields:
      additionalProperties: Additional properties of type ErrorsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ErrorsValue object.

      Fields:
        key: Name of the additional property.
        value: A Status attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('Status', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    errors = _messages.MessageField('ErrorsValue', 1)
    state = _messages.EnumField('StateValueValuesEnum', 2)