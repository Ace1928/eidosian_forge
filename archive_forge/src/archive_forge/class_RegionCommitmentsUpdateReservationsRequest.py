from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionCommitmentsUpdateReservationsRequest(_messages.Message):
    """A RegionCommitmentsUpdateReservationsRequest object.

  Fields:
    reservations: A list of two reservations to transfer GPUs and local SSD
      between.
  """
    reservations = _messages.MessageField('Reservation', 1, repeated=True)