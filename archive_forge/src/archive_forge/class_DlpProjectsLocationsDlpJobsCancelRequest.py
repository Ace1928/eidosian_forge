from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsDlpJobsCancelRequest(_messages.Message):
    """A DlpProjectsLocationsDlpJobsCancelRequest object.

  Fields:
    googlePrivacyDlpV2CancelDlpJobRequest: A
      GooglePrivacyDlpV2CancelDlpJobRequest resource to be passed as the
      request body.
    name: Required. The name of the DlpJob resource to be cancelled.
  """
    googlePrivacyDlpV2CancelDlpJobRequest = _messages.MessageField('GooglePrivacyDlpV2CancelDlpJobRequest', 1)
    name = _messages.StringField(2, required=True)