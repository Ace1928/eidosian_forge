from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetOtherCloudConnectionsCreateRequest(_messages.Message):
    """A CloudassetOtherCloudConnectionsCreateRequest object.

  Fields:
    otherCloudConnection: A OtherCloudConnection resource to be passed as the
      request body.
    otherCloudConnectionId: Required. The ID to use for the other-cloud
      connection, which will become the final component of the other-cloud
      connection's resource name. Currently only "aws" is allowed as the
      other_cloud_connection_id.
    parent: Required. The parent resource where this connection will be
      created. It can only be an organization number (such as
      "organizations/123") for now. Format:
      organizations/{organization_number} (e.g., "organizations/123456").
  """
    otherCloudConnection = _messages.MessageField('OtherCloudConnection', 1)
    otherCloudConnectionId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)