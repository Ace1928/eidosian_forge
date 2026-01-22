from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritypostureOrganizationsLocationsPosturesCreateRequest(_messages.Message):
    """A SecuritypostureOrganizationsLocationsPosturesCreateRequest object.

  Fields:
    parent: Required. Value for parent.
    posture: A Posture resource to be passed as the request body.
    postureId: Required. User provided identifier. It should be unique in
      scope of an Organization and location.
  """
    parent = _messages.StringField(1, required=True)
    posture = _messages.MessageField('Posture', 2)
    postureId = _messages.StringField(3)