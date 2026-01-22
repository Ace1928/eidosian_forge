from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2UpdateInspectTemplateRequest(_messages.Message):
    """Request message for UpdateInspectTemplate.

  Fields:
    inspectTemplate: New InspectTemplate value.
    updateMask: Mask to control which fields get updated.
  """
    inspectTemplate = _messages.MessageField('GooglePrivacyDlpV2InspectTemplate', 1)
    updateMask = _messages.StringField(2)