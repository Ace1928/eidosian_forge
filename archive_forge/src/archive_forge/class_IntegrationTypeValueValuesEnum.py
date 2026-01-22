from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IntegrationTypeValueValuesEnum(_messages.Enum):
    """Required. Describes how this key is integrated with the website.

    Values:
      INTEGRATION_TYPE_UNSPECIFIED: Default type that indicates this enum
        hasn't been specified. This is not a valid IntegrationType, one of the
        other types must be specified instead.
      SCORE: Only used to produce scores. It doesn't display the "I'm not a
        robot" checkbox and never shows captcha challenges.
      CHECKBOX: Displays the "I'm not a robot" checkbox and may show captcha
        challenges after it is checked.
      INVISIBLE: Doesn't display the "I'm not a robot" checkbox, but may show
        captcha challenges after risk analysis.
    """
    INTEGRATION_TYPE_UNSPECIFIED = 0
    SCORE = 1
    CHECKBOX = 2
    INVISIBLE = 3