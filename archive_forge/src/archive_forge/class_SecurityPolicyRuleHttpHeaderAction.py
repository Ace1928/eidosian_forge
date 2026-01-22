from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleHttpHeaderAction(_messages.Message):
    """A SecurityPolicyRuleHttpHeaderAction object.

  Fields:
    requestHeadersToAdds: The list of request headers to add or overwrite if
      they're already present.
  """
    requestHeadersToAdds = _messages.MessageField('SecurityPolicyRuleHttpHeaderActionHttpHeaderOption', 1, repeated=True)