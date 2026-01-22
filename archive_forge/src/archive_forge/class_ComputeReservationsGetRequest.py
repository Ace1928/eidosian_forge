from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeReservationsGetRequest(_messages.Message):
    """A ComputeReservationsGetRequest object.

  Fields:
    project: Project ID for this request.
    reservation: Name of the reservation to retrieve.
    zone: Name of the zone for this request.
  """
    project = _messages.StringField(1, required=True)
    reservation = _messages.StringField(2, required=True)
    zone = _messages.StringField(3, required=True)