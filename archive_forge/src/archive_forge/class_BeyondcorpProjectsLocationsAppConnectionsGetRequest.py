from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppConnectionsGetRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppConnectionsGetRequest object.

  Fields:
    name: Required. BeyondCorp AppConnection name using the form: `projects/{p
      roject_id}/locations/{location_id}/appConnections/{app_connection_id}`
  """
    name = _messages.StringField(1, required=True)