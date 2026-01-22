from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityPolicyRuleMatcherExprOptionsRecaptchaOptions(_messages.Message):
    """A SecurityPolicyRuleMatcherExprOptionsRecaptchaOptions object.

  Fields:
    actionTokenSiteKeys: A list of site keys to be used during the validation
      of reCAPTCHA action-tokens. The provided site keys need to be created
      from reCAPTCHA API under the same project where the security policy is
      created.
    sessionTokenSiteKeys: A list of site keys to be used during the validation
      of reCAPTCHA session-tokens. The provided site keys need to be created
      from reCAPTCHA API under the same project where the security policy is
      created.
  """
    actionTokenSiteKeys = _messages.StringField(1, repeated=True)
    sessionTokenSiteKeys = _messages.StringField(2, repeated=True)