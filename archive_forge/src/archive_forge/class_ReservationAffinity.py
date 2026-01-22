from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReservationAffinity(_messages.Message):
    """[ReservationAffinity](https://cloud.google.com/compute/docs/instances/re
  serving-zonal-resources) is the configuration of desired reservation which
  instances could take capacity from.

  Enums:
    ConsumeReservationTypeValueValuesEnum: Corresponds to the type of
      reservation consumption.

  Fields:
    consumeReservationType: Corresponds to the type of reservation
      consumption.
    key: Corresponds to the label key of a reservation resource. To target a
      SPECIFIC_RESERVATION by name, specify
      "compute.googleapis.com/reservation-name" as the key and specify the
      name of your reservation as its value.
    values: Corresponds to the label value(s) of reservation resource(s).
  """

    class ConsumeReservationTypeValueValuesEnum(_messages.Enum):
        """Corresponds to the type of reservation consumption.

    Values:
      UNSPECIFIED: Default value. This should not be used.
      NO_RESERVATION: Do not consume from any reserved capacity.
      ANY_RESERVATION: Consume any reservation available.
      SPECIFIC_RESERVATION: Must consume from a specific reservation. Must
        specify key value fields for specifying the reservations.
    """
        UNSPECIFIED = 0
        NO_RESERVATION = 1
        ANY_RESERVATION = 2
        SPECIFIC_RESERVATION = 3
    consumeReservationType = _messages.EnumField('ConsumeReservationTypeValueValuesEnum', 1)
    key = _messages.StringField(2)
    values = _messages.StringField(3, repeated=True)