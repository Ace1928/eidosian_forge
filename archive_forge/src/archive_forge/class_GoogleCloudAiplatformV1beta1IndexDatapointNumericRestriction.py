from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1IndexDatapointNumericRestriction(_messages.Message):
    """This field allows restricts to be based on numeric comparisons rather
  than categorical tokens.

  Enums:
    OpValueValuesEnum: This MUST be specified for queries and must NOT be
      specified for datapoints.

  Fields:
    namespace: The namespace of this restriction. e.g.: cost.
    op: This MUST be specified for queries and must NOT be specified for
      datapoints.
    valueDouble: Represents 64 bit float.
    valueFloat: Represents 32 bit float.
    valueInt: Represents 64 bit integer.
  """

    class OpValueValuesEnum(_messages.Enum):
        """This MUST be specified for queries and must NOT be specified for
    datapoints.

    Values:
      OPERATOR_UNSPECIFIED: Default value of the enum.
      LESS: Datapoints are eligible iff their value is < the query's.
      LESS_EQUAL: Datapoints are eligible iff their value is <= the query's.
      EQUAL: Datapoints are eligible iff their value is == the query's.
      GREATER_EQUAL: Datapoints are eligible iff their value is >= the
        query's.
      GREATER: Datapoints are eligible iff their value is > the query's.
      NOT_EQUAL: Datapoints are eligible iff their value is != the query's.
    """
        OPERATOR_UNSPECIFIED = 0
        LESS = 1
        LESS_EQUAL = 2
        EQUAL = 3
        GREATER_EQUAL = 4
        GREATER = 5
        NOT_EQUAL = 6
    namespace = _messages.StringField(1)
    op = _messages.EnumField('OpValueValuesEnum', 2)
    valueDouble = _messages.FloatField(3)
    valueFloat = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    valueInt = _messages.IntegerField(5)