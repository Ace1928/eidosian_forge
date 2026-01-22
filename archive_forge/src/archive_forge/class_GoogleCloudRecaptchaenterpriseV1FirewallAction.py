from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FirewallAction(_messages.Message):
    """An individual action. Each action represents what to do if a policy
  matches.

  Fields:
    allow: The user request did not match any policy and should be allowed
      access to the requested resource.
    block: This action will deny access to a given page. The user will get an
      HTTP error code.
    includeRecaptchaScript: This action will inject reCAPTCHA JavaScript code
      into the HTML page returned by the site backend.
    redirect: This action will redirect the request to a ReCaptcha
      interstitial to attach a token.
    setHeader: This action will set a custom header but allow the request to
      continue to the customer backend.
    substitute: This action will transparently serve a different page to an
      offending user.
  """
    allow = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallActionAllowAction', 1)
    block = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallActionBlockAction', 2)
    includeRecaptchaScript = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallActionIncludeRecaptchaScriptAction', 3)
    redirect = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallActionRedirectAction', 4)
    setHeader = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallActionSetHeaderAction', 5)
    substitute = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FirewallActionSubstituteAction', 6)