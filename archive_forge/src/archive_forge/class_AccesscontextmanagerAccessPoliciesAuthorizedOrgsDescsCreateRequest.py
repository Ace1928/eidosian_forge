from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsCreateRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAuthorizedOrgsDescsCreateRequest
  object.

  Fields:
    authorizedOrgsDesc: A AuthorizedOrgsDesc resource to be passed as the
      request body.
    parent: Required. Resource name for the access policy which owns this
      Authorized Orgs Desc. Format: `accessPolicies/{policy_id}`
  """
    authorizedOrgsDesc = _messages.MessageField('AuthorizedOrgsDesc', 1)
    parent = _messages.StringField(2, required=True)