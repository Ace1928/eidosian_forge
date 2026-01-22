from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectJobConfig(_messages.Message):
    """Controls what and how to inspect for findings.

  Fields:
    actions: Actions to execute at the completion of the job.
    inspectConfig: How and what to scan for.
    inspectTemplateName: If provided, will be used as the default for all
      values in InspectConfig. `inspect_config` will be merged into the values
      persisted as part of the template.
    storageConfig: The data to scan.
  """
    actions = _messages.MessageField('GooglePrivacyDlpV2Action', 1, repeated=True)
    inspectConfig = _messages.MessageField('GooglePrivacyDlpV2InspectConfig', 2)
    inspectTemplateName = _messages.StringField(3)
    storageConfig = _messages.MessageField('GooglePrivacyDlpV2StorageConfig', 4)