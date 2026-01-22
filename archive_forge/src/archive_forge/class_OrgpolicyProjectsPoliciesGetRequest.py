from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyProjectsPoliciesGetRequest(_messages.Message):
    """A OrgpolicyProjectsPoliciesGetRequest object.

  Fields:
    name: Required. Resource name of the policy. See Policy for naming
      requirements.
  """
    name = _messages.StringField(1, required=True)