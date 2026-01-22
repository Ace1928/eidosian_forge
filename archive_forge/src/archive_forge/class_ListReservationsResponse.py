from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListReservationsResponse(_messages.Message):
    """Response for ListReservations.

  Fields:
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page of results. If this field is omitted, there are no more
      results.
    reservations: The list of reservation in the requested parent. The order
      of the reservations is unspecified.
  """
    nextPageToken = _messages.StringField(1)
    reservations = _messages.MessageField('Reservation', 2, repeated=True)