from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DatabaseResourceRegexes(_messages.Message):
    """A collection of regular expressions to determine what database resources
  to match against.

  Fields:
    patterns: A group of regular expression patterns to match against one or
      more database resources. Maximum of 100 entries. The sum of all regular
      expression's length can't exceed 10 KiB.
  """
    patterns = _messages.MessageField('GooglePrivacyDlpV2DatabaseResourceRegex', 1, repeated=True)