from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ValidateMessageRequest(_messages.Message):
    """Request for the `ValidateMessage` method.

  Enums:
    EncodingValueValuesEnum: The encoding expected for messages

  Fields:
    encoding: The encoding expected for messages
    message: Message to validate against the provided `schema_spec`.
    name: Name of the schema against which to validate. Format is
      `projects/{project}/schemas/{schema}`.
    schema: Ad-hoc schema against which to validate
  """

    class EncodingValueValuesEnum(_messages.Enum):
        """The encoding expected for messages

    Values:
      ENCODING_UNSPECIFIED: Unspecified
      JSON: JSON encoding
      BINARY: Binary encoding, as defined by the schema type. For some schema
        types, binary encoding may not be available.
    """
        ENCODING_UNSPECIFIED = 0
        JSON = 1
        BINARY = 2
    encoding = _messages.EnumField('EncodingValueValuesEnum', 1)
    message = _messages.BytesField(2)
    name = _messages.StringField(3)
    schema = _messages.MessageField('Schema', 4)