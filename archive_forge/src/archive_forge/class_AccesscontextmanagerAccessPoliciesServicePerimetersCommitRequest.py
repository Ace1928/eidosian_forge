from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesServicePerimetersCommitRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesServicePerimetersCommitRequest
  object.

  Fields:
    commitServicePerimetersRequest: A CommitServicePerimetersRequest resource
      to be passed as the request body.
    parent: Required. Resource name for the parent Access Policy which owns
      all Service Perimeters in scope for the commit operation. Format:
      `accessPolicies/{policy_id}`
  """
    commitServicePerimetersRequest = _messages.MessageField('CommitServicePerimetersRequest', 1)
    parent = _messages.StringField(2, required=True)