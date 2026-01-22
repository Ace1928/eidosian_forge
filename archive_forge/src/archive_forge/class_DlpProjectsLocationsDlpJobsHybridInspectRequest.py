from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DlpProjectsLocationsDlpJobsHybridInspectRequest(_messages.Message):
    """A DlpProjectsLocationsDlpJobsHybridInspectRequest object.

  Fields:
    googlePrivacyDlpV2HybridInspectDlpJobRequest: A
      GooglePrivacyDlpV2HybridInspectDlpJobRequest resource to be passed as
      the request body.
    name: Required. Resource name of the job to execute a hybrid inspect on,
      for example `projects/dlp-test-project/dlpJob/53234423`.
  """
    googlePrivacyDlpV2HybridInspectDlpJobRequest = _messages.MessageField('GooglePrivacyDlpV2HybridInspectDlpJobRequest', 1)
    name = _messages.StringField(2, required=True)