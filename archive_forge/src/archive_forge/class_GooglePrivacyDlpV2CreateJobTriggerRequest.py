from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CreateJobTriggerRequest(_messages.Message):
    """Request message for CreateJobTrigger.

  Fields:
    jobTrigger: Required. The JobTrigger to create.
    locationId: Deprecated. This field has no effect.
    triggerId: The trigger id can contain uppercase and lowercase letters,
      numbers, and hyphens; that is, it must match the regular expression:
      `[a-zA-Z\\d-_]+`. The maximum length is 100 characters. Can be empty to
      allow the system to generate one.
  """
    jobTrigger = _messages.MessageField('GooglePrivacyDlpV2JobTrigger', 1)
    locationId = _messages.StringField(2)
    triggerId = _messages.StringField(3)