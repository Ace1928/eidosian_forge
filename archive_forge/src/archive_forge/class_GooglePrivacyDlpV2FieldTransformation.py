from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2FieldTransformation(_messages.Message):
    """The transformation to apply to the field.

  Fields:
    condition: Only apply the transformation if the condition evaluates to
      true for the given `RecordCondition`. The conditions are allowed to
      reference fields that are not used in the actual transformation. Example
      Use Cases: - Apply a different bucket transformation to an age column if
      the zip code column for the same record is within a specific range. -
      Redact a field if the date of birth field is greater than 85.
    fields: Required. Input field(s) to apply the transformation to. When you
      have columns that reference their position within a list, omit the index
      from the FieldId. FieldId name matching ignores the index. For example,
      instead of "contact.nums[0].type", use "contact.nums.type".
    infoTypeTransformations: Treat the contents of the field as free text, and
      selectively transform content that matches an `InfoType`.
    primitiveTransformation: Apply the transformation to the entire field.
  """
    condition = _messages.MessageField('GooglePrivacyDlpV2RecordCondition', 1)
    fields = _messages.MessageField('GooglePrivacyDlpV2FieldId', 2, repeated=True)
    infoTypeTransformations = _messages.MessageField('GooglePrivacyDlpV2InfoTypeTransformations', 3)
    primitiveTransformation = _messages.MessageField('GooglePrivacyDlpV2PrimitiveTransformation', 4)