from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2alpha1Restrictions(_messages.Message):
    """Restrictions for all types of API Keys.

  Fields:
    androidKeyRestrictions: Key restriction that are specific to android keys.
      Android apps
    apiTargets: A restriction for a specific service and optionally one or
      multiple specific methods. Requests will be allowed if they match any of
      these restrictions. If no restrictions are specified, all targets are
      allowed.
    browserKeyRestrictions: Key restrictions that are specific to browser
      keys. Referer
    iosKeyRestrictions: Key restriction that are specific to iOS keys. IOS app
      id
    serverKeyRestrictions: Key restrictions that are specific to server keys.
      Allowed ips
  """
    androidKeyRestrictions = _messages.MessageField('V2alpha1AndroidKeyRestrictions', 1)
    apiTargets = _messages.MessageField('V2alpha1ApiTarget', 2, repeated=True)
    browserKeyRestrictions = _messages.MessageField('V2alpha1BrowserKeyRestrictions', 3)
    iosKeyRestrictions = _messages.MessageField('V2alpha1IosKeyRestrictions', 4)
    serverKeyRestrictions = _messages.MessageField('V2alpha1ServerKeyRestrictions', 5)