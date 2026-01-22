from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsBindingsCreateRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsBindingsCreateRequest object.

  Fields:
    membershipBinding: A MembershipBinding resource to be passed as the
      request body.
    membershipBindingId: Required. The ID to use for the MembershipBinding.
    parent: Required. The parent (project and location) where the
      MembershipBinding will be created. Specified in the format
      `projects/*/locations/*/memberships/*`.
  """
    membershipBinding = _messages.MessageField('MembershipBinding', 1)
    membershipBindingId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)