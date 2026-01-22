from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ImportEntriesMetadata(_messages.Message):
    """Metadata message for long-running operation returned by the
  ImportEntries.

  Enums:
    StateValueValuesEnum: State of the import operation.

  Fields:
    errors: Partial errors that are encountered during the ImportEntries
      operation. There is no guarantee that all the encountered errors are
      reported. However, if no errors are reported, it means that no errors
      were encountered.
    state: State of the import operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """State of the import operation.

    Values:
      IMPORT_STATE_UNSPECIFIED: Default value. This value is unused.
      IMPORT_QUEUED: The dump with entries has been queued for import.
      IMPORT_IN_PROGRESS: The import of entries is in progress.
      IMPORT_DONE: The import of entries has been finished.
      IMPORT_OBSOLETE: The import of entries has been abandoned in favor of a
        newer request.
    """
        IMPORT_STATE_UNSPECIFIED = 0
        IMPORT_QUEUED = 1
        IMPORT_IN_PROGRESS = 2
        IMPORT_DONE = 3
        IMPORT_OBSOLETE = 4
    errors = _messages.MessageField('Status', 1, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 2)