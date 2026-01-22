from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2BrowserKeyRestrictions(_messages.Message):
    """The HTTP referrers (websites) that are allowed to use the key.

  Fields:
    allowedReferrers: A list of regular expressions for the referrer URLs that
      are allowed to make API calls with this key.
  """
    allowedReferrers = _messages.StringField(1, repeated=True)