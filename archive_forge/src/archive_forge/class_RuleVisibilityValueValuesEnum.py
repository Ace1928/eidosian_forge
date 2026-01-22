from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuleVisibilityValueValuesEnum(_messages.Enum):
    """Rule visibility can be one of the following: STANDARD - opaque rules.
    (default) PREMIUM - transparent rules. This field is only supported in
    Global Security Policies of type CLOUD_ARMOR.

    Values:
      PREMIUM: <no description>
      STANDARD: <no description>
    """
    PREMIUM = 0
    STANDARD = 1