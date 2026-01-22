from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkmanagementProjectsLocationsGlobalConnectivityTestsDeleteRequest(_messages.Message):
    """A NetworkmanagementProjectsLocationsGlobalConnectivityTestsDeleteRequest
  object.

  Fields:
    name: Required. Connectivity Test resource name using the form:
      `projects/{project_id}/locations/global/connectivityTests/{test_id}`
  """
    name = _messages.StringField(1, required=True)