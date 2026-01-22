from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkehubProjectsLocationsMembershipsBindingsPatchRequest(_messages.Message):
    """A GkehubProjectsLocationsMembershipsBindingsPatchRequest object.

  Fields:
    membershipBinding: A MembershipBinding resource to be passed as the
      request body.
    name: The resource name for the membershipbinding itself `projects/{projec
      t}/locations/{location}/memberships/{membership}/bindings/{membershipbin
      ding}`
    updateMask: Required. The fields to be updated.
  """
    membershipBinding = _messages.MessageField('MembershipBinding', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)