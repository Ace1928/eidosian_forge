from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetOtherCloudConnectionsGetRequest(_messages.Message):
    """A CloudassetOtherCloudConnectionsGetRequest object.

  Fields:
    name: Required. The name of the other-cloud connection to retrieve.
      Format: organizations/{organization_number}/otherCloudConnections/{other
      _cloud_connection_id}
  """
    name = _messages.StringField(1, required=True)