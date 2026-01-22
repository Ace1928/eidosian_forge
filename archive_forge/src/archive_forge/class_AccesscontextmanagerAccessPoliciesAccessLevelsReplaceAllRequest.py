from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesAccessLevelsReplaceAllRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesAccessLevelsReplaceAllRequest
  object.

  Fields:
    parent: Required. Resource name for the access policy which owns these
      Access Levels. Format: `accessPolicies/{policy_id}`
    replaceAccessLevelsRequest: A ReplaceAccessLevelsRequest resource to be
      passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    replaceAccessLevelsRequest = _messages.MessageField('ReplaceAccessLevelsRequest', 2)