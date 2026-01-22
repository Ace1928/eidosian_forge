from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1WafSettings(_messages.Message):
    """Settings specific to keys that can be used for WAF (Web Application
  Firewall).

  Enums:
    WafFeatureValueValuesEnum: Required. The WAF feature for which this key is
      enabled.
    WafServiceValueValuesEnum: Required. The WAF service that uses this key.

  Fields:
    wafFeature: Required. The WAF feature for which this key is enabled.
    wafService: Required. The WAF service that uses this key.
  """

    class WafFeatureValueValuesEnum(_messages.Enum):
        """Required. The WAF feature for which this key is enabled.

    Values:
      WAF_FEATURE_UNSPECIFIED: Undefined feature.
      CHALLENGE_PAGE: Redirects suspicious traffic to reCAPTCHA.
      SESSION_TOKEN: Use reCAPTCHA session-tokens to protect the whole user
        session on the site's domain.
      ACTION_TOKEN: Use reCAPTCHA action-tokens to protect user actions.
      EXPRESS: Use reCAPTCHA WAF express protection to protect any content
        other than web pages, like APIs and IoT devices.
    """
        WAF_FEATURE_UNSPECIFIED = 0
        CHALLENGE_PAGE = 1
        SESSION_TOKEN = 2
        ACTION_TOKEN = 3
        EXPRESS = 4

    class WafServiceValueValuesEnum(_messages.Enum):
        """Required. The WAF service that uses this key.

    Values:
      WAF_SERVICE_UNSPECIFIED: Undefined WAF
      CA: Cloud Armor
      FASTLY: Fastly
      CLOUDFLARE: Cloudflare
      AKAMAI: Akamai
    """
        WAF_SERVICE_UNSPECIFIED = 0
        CA = 1
        FASTLY = 2
        CLOUDFLARE = 3
        AKAMAI = 4
    wafFeature = _messages.EnumField('WafFeatureValueValuesEnum', 1)
    wafService = _messages.EnumField('WafServiceValueValuesEnum', 2)