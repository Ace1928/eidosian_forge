from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1ReviewDocumentRequest(_messages.Message):
    """Request message for the ReviewDocument method.

  Enums:
    PriorityValueValuesEnum: The priority of the human review task.

  Fields:
    documentSchema: The document schema of the human review task.
    enableSchemaValidation: Whether the validation should be performed on the
      ad-hoc review request.
    inlineDocument: An inline document proto.
    priority: The priority of the human review task.
  """

    class PriorityValueValuesEnum(_messages.Enum):
        """The priority of the human review task.

    Values:
      DEFAULT: The default priority level.
      URGENT: The urgent priority level. The labeling manager should allocate
        labeler resource to the urgent task queue to respect this priority
        level.
    """
        DEFAULT = 0
        URGENT = 1
    documentSchema = _messages.MessageField('GoogleCloudDocumentaiV1DocumentSchema', 1)
    enableSchemaValidation = _messages.BooleanField(2)
    inlineDocument = _messages.MessageField('GoogleCloudDocumentaiV1Document', 3)
    priority = _messages.EnumField('PriorityValueValuesEnum', 4)