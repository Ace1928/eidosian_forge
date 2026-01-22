from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentifierHelper(_messages.Message):
    """Helps in identifying the underlying product. This should be treated like
  a one-of field. Only one field should be set in this proto. This is a
  workaround because spanner indexes on one-of fields restrict addition and
  deletion of fields.

  Enums:
    FieldValueValuesEnum: The field that is set in the API proto.

  Fields:
    field: The field that is set in the API proto.
    genericUri: Contains a URI which is vendor-specific. Example: The artifact
      repository URL of an image.
  """

    class FieldValueValuesEnum(_messages.Enum):
        """The field that is set in the API proto.

    Values:
      IDENTIFIER_HELPER_FIELD_UNSPECIFIED: The helper isn't set.
      GENERIC_URI: The generic_uri one-of field is set.
    """
        IDENTIFIER_HELPER_FIELD_UNSPECIFIED = 0
        GENERIC_URI = 1
    field = _messages.EnumField('FieldValueValuesEnum', 1)
    genericUri = _messages.StringField(2)