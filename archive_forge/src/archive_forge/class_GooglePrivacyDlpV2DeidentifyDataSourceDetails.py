from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DeidentifyDataSourceDetails(_messages.Message):
    """The results of a Deidentify action from an inspect job.

  Fields:
    deidentifyStats: Stats about the de-identification operation.
    requestedOptions: De-identification config used for the request.
  """
    deidentifyStats = _messages.MessageField('GooglePrivacyDlpV2DeidentifyDataSourceStats', 1)
    requestedOptions = _messages.MessageField('GooglePrivacyDlpV2RequestedDeidentifyOptions', 2)