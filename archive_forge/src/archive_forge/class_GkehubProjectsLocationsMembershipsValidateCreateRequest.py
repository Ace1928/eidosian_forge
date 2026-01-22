from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsValidateCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsValidateCreateRequest object.

  Fields:
    parent: Required. The parent (project and location) where the Memberships
      will be created. Specified in the format `projects/*/locations/*`.
    validateCreateMembershipRequest: A ValidateCreateMembershipRequest
      resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    validateCreateMembershipRequest = _messages.MessageField('ValidateCreateMembershipRequest', 2)