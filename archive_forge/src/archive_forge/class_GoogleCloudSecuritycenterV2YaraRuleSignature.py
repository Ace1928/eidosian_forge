from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2YaraRuleSignature(_messages.Message):
    """A signature corresponding to a YARA rule.

  Fields:
    yaraRule: The name of the YARA rule.
  """
    yaraRule = _messages.StringField(1)