from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReservationUsageValueListEntry(_messages.Message):
    """Job resource usage breakdown by reservation.

    Fields:
      name: Reservation name or "unreserved" for on-demand resources usage.
      slotMs: Total slot milliseconds used by the reservation for a particular
        job.
    """
    name = _messages.StringField(1)
    slotMs = _messages.IntegerField(2)