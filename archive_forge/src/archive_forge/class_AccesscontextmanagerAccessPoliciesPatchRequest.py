from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccesscontextmanagerAccessPoliciesPatchRequest(_messages.Message):
    """A AccesscontextmanagerAccessPoliciesPatchRequest object.

  Fields:
    accessPolicy: A AccessPolicy resource to be passed as the request body.
    name: Resource name of the `AccessPolicy`. Format:
      `accessPolicies/{access_policy}`
    updateMask: Required. Mask to control which fields get updated. Must be
      non-empty.
  """
    accessPolicy = _messages.MessageField('AccessPolicy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)