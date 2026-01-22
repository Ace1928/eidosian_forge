from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubliteAdminProjectsLocationsReservationsGetRequest(_messages.Message):
    """A PubsubliteAdminProjectsLocationsReservationsGetRequest object.

  Fields:
    name: Required. The name of the reservation whose configuration to return.
      Structured like: projects/{project_number}/locations/{location}/reservat
      ions/{reservation_id}
  """
    name = _messages.StringField(1, required=True)