from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1ServiceIdentity(_messages.Message):
    """A service identity in the Identity and Access Management API.

  Fields:
    email: The email address of the service identity.
    name: P4 service identity resource name.  An example name would be: `servi
      ces/serviceconsumermanagement.googleapis.com/projects/123/serviceIdentit
      ies/default`
    tag: The P4 service identity configuration tag. This must be defined in
      activation_grants. If not specified when creating the account, the tag
      is set to "default".
    uniqueId: The unique and stable id of the service identity.
  """
    email = _messages.StringField(1)
    name = _messages.StringField(2)
    tag = _messages.StringField(3)
    uniqueId = _messages.StringField(4)