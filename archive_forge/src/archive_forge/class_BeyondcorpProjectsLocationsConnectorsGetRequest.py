from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsConnectorsGetRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsConnectorsGetRequest object.

  Fields:
    name: Required. BeyondCorp Connector name using the form:
      `projects/{project_id}/locations/{location_id}/connectors/{connector_id}
      `
  """
    name = _messages.StringField(1, required=True)