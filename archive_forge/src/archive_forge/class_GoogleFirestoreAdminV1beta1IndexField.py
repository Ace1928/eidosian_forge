from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta1IndexField(_messages.Message):
    """A field of an index.

  Enums:
    ModeValueValuesEnum: The field's mode.

  Fields:
    fieldPath: The path of the field. Must match the field path specification
      described by google.firestore.v1beta1.Document.fields. Special field
      path `__name__` may be used by itself or at the end of a path.
      `__type__` may be used only at the end of path.
    mode: The field's mode.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """The field's mode.

    Values:
      MODE_UNSPECIFIED: The mode is unspecified.
      ASCENDING: The field's values are indexed so as to support sequencing in
        ascending order and also query by <, >, <=, >=, and =.
      DESCENDING: The field's values are indexed so as to support sequencing
        in descending order and also query by <, >, <=, >=, and =.
      ARRAY_CONTAINS: The field's array values are indexed so as to support
        membership using ARRAY_CONTAINS queries.
    """
        MODE_UNSPECIFIED = 0
        ASCENDING = 1
        DESCENDING = 2
        ARRAY_CONTAINS = 3
    fieldPath = _messages.StringField(1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)