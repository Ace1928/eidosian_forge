from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsReservationsCreateRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsReservationsCreateRequest object.

  Fields:
    parent: Required. The parent location in which to create the reservation.
      Structured like `projects/{project_number}/locations/{location}`.
    reservation: A Reservation resource to be passed as the request body.
    reservationId: Required. The ID to use for the reservation, which will
      become the final component of the reservation's name. This value is
      structured like: `my-reservation-name`.
  """
    parent = _messages.StringField(1, required=True)
    reservation = _messages.MessageField('Reservation', 2)
    reservationId = _messages.StringField(3)