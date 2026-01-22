from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReservationConfig(_messages.Message):
    """The settings for this topic's Reservation usage.

  Fields:
    throughputReservation: The Reservation to use for this topic's throughput
      capacity. Structured like: projects/{project_number}/locations/{location
      }/reservations/{reservation_id}
  """
    throughputReservation = _messages.StringField(1)