from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllocationAffinity(_messages.Message):
    """Specifies the reservations that this instance can consume from.

  Enums:
    ConsumeReservationTypeValueValuesEnum: Optional. Specifies the type of
      reservation from which this instance can consume

  Fields:
    consumeReservationType: Optional. Specifies the type of reservation from
      which this instance can consume
    key: Optional. Corresponds to the label key of a reservation resource.
    values: Optional. Corresponds to the label values of a reservation
      resource.
  """

    class ConsumeReservationTypeValueValuesEnum(_messages.Enum):
        """Optional. Specifies the type of reservation from which this instance
    can consume

    Values:
      TYPE_UNSPECIFIED: Default value. This value is unused.
      NO_ALLOCATION: Do not consume from any allocated capacity.
      ANY_ALLOCATION: Consume any allocation available.
      SPECIFIC_ALLOCATION: Must consume from a specific reservation. Must
        specify key value fields for specifying the reservations.
    """
        TYPE_UNSPECIFIED = 0
        NO_ALLOCATION = 1
        ANY_ALLOCATION = 2
        SPECIFIC_ALLOCATION = 3
    consumeReservationType = _messages.EnumField('ConsumeReservationTypeValueValuesEnum', 1)
    key = _messages.StringField(2)
    values = _messages.StringField(3, repeated=True)