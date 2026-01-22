from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGsuiteaddonsV1Deployment(_messages.Message):
    """A Google Workspace Add-on deployment

  Fields:
    addOns: The Google Workspace Add-on configuration.
    etag: This value is computed by the server based on the version of the
      deployment in storage, and may be sent on update and delete requests to
      ensure the client has an up-to-date value before proceeding.
    name: The deployment resource name. Example:
      `projects/123/deployments/my_deployment`.
    oauthScopes: The list of Google OAuth scopes for which to request consent
      from the end user before executing an add-on endpoint.
  """
    addOns = _messages.MessageField('GoogleCloudGsuiteaddonsV1AddOns', 1)
    etag = _messages.StringField(2)
    name = _messages.StringField(3)
    oauthScopes = _messages.StringField(4, repeated=True)