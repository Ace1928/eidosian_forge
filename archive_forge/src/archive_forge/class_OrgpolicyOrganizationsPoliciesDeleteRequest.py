from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyOrganizationsPoliciesDeleteRequest(_messages.Message):
    """A OrgpolicyOrganizationsPoliciesDeleteRequest object.

  Fields:
    etag: Optional. The current etag of policy. If an etag is provided and
      does not match the current etag of the policy, deletion will be blocked
      and an ABORTED error will be returned.
    name: Required. Name of the policy to delete. See the policy entry for
      naming rules.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)