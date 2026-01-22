from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationLocation(_messages.Message):
    """Specifies the location of a transformation.

  Enums:
    ContainerTypeValueValuesEnum: Information about the functionality of the
      container where this finding occurred, if available.

  Fields:
    containerType: Information about the functionality of the container where
      this finding occurred, if available.
    findingId: For infotype transformations, link to the corresponding
      findings ID so that location information does not need to be duplicated.
      Each findings ID correlates to an entry in the findings output table,
      this table only gets created when users specify to save findings (add
      the save findings action to the request).
    recordTransformation: For record transformations, provide a field and
      container information.
  """

    class ContainerTypeValueValuesEnum(_messages.Enum):
        """Information about the functionality of the container where this
    finding occurred, if available.

    Values:
      TRANSFORM_UNKNOWN_CONTAINER: Unused.
      TRANSFORM_BODY: Body of a file.
      TRANSFORM_METADATA: Metadata for a file.
      TRANSFORM_TABLE: A table.
    """
        TRANSFORM_UNKNOWN_CONTAINER = 0
        TRANSFORM_BODY = 1
        TRANSFORM_METADATA = 2
        TRANSFORM_TABLE = 3
    containerType = _messages.EnumField('ContainerTypeValueValuesEnum', 1)
    findingId = _messages.StringField(2)
    recordTransformation = _messages.MessageField('GooglePrivacyDlpV2RecordTransformation', 3)