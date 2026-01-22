from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsConnectionsGetRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsConnectionsGetRequest object.

  Fields:
    name: Required. BeyondCorp Connection name using the form: `projects/{proj
      ect_id}/locations/{location_id}/connections/{connection_id}`
  """
    name = _messages.StringField(1, required=True)