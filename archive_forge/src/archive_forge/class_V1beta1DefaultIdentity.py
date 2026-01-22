from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1beta1DefaultIdentity(_messages.Message):
    """A default identity in the Identity and Access Management API.

  Fields:
    email: The email address of the default identity.
    name: Default identity resource name.  An example name would be: `services
      /serviceconsumermanagement.googleapis.com/projects/123/defaultIdentity`
    uniqueId: The unique and stable id of the default identity.
  """
    email = _messages.StringField(1)
    name = _messages.StringField(2)
    uniqueId = _messages.StringField(3)