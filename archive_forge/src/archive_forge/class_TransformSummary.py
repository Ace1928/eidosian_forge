from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransformSummary(_messages.Message):
    """Description of the type, names/ids, and input/outputs for a transform.

  Enums:
    KindValueValuesEnum: Type of transform.

  Fields:
    displayData: Transform-specific display data.
    id: SDK generated id of this transform instance.
    inputCollectionName: User names for all collection inputs to this
      transform.
    kind: Type of transform.
    name: User provided name for this transform instance.
    outputCollectionName: User names for all collection outputs to this
      transform.
  """

    class KindValueValuesEnum(_messages.Enum):
        """Type of transform.

    Values:
      UNKNOWN_KIND: Unrecognized transform type.
      PAR_DO_KIND: ParDo transform.
      GROUP_BY_KEY_KIND: Group By Key transform.
      FLATTEN_KIND: Flatten transform.
      READ_KIND: Read transform.
      WRITE_KIND: Write transform.
      CONSTANT_KIND: Constructs from a constant value, such as with Create.of.
      SINGLETON_KIND: Creates a Singleton view of a collection.
      SHUFFLE_KIND: Opening or closing a shuffle session, often as part of a
        GroupByKey.
    """
        UNKNOWN_KIND = 0
        PAR_DO_KIND = 1
        GROUP_BY_KEY_KIND = 2
        FLATTEN_KIND = 3
        READ_KIND = 4
        WRITE_KIND = 5
        CONSTANT_KIND = 6
        SINGLETON_KIND = 7
        SHUFFLE_KIND = 8
    displayData = _messages.MessageField('DisplayData', 1, repeated=True)
    id = _messages.StringField(2)
    inputCollectionName = _messages.StringField(3, repeated=True)
    kind = _messages.EnumField('KindValueValuesEnum', 4)
    name = _messages.StringField(5)
    outputCollectionName = _messages.StringField(6, repeated=True)