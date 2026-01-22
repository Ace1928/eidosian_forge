from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaListBrowserDlpRulesResponse(_messages.Message):
    """Message for response to listing BrowserDlpRules.

  Fields:
    browserDlpRules: The list of BrowserDlpRule objects.
  """
    browserDlpRules = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaBrowserDlpRule', 1, repeated=True)