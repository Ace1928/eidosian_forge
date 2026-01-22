from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGsuiteaddonsV1Authorization(_messages.Message):
    """The authorization information used when invoking deployment endpoints.

  Fields:
    name: The canonical full name of this resource. Example:
      `projects/123/authorization`
    oauthClientId: The OAuth client ID used to obtain OAuth access tokens for
      a user on the add-on's behalf.
    serviceAccountEmail: The email address of the service account used to
      authenticate requests to add-on callback endpoints.
  """
    name = _messages.StringField(1)
    oauthClientId = _messages.StringField(2)
    serviceAccountEmail = _messages.StringField(3)