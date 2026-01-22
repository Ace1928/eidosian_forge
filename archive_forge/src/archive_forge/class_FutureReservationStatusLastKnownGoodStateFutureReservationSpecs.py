from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FutureReservationStatusLastKnownGoodStateFutureReservationSpecs(_messages.Message):
    """The properties of the last known good state for the Future Reservation.

  Fields:
    shareSettings: [Output Only] The previous share settings of the Future
      Reservation.
    specificSkuProperties: [Output Only] The previous instance related
      properties of the Future Reservation.
    timeWindow: [Output Only] The previous time window of the Future
      Reservation.
  """
    shareSettings = _messages.MessageField('ShareSettings', 1)
    specificSkuProperties = _messages.MessageField('FutureReservationSpecificSKUProperties', 2)
    timeWindow = _messages.MessageField('FutureReservationTimeWindow', 3)