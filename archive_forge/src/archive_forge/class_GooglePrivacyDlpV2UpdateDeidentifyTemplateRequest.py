from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2UpdateDeidentifyTemplateRequest(_messages.Message):
    """Request message for UpdateDeidentifyTemplate.

  Fields:
    deidentifyTemplate: New DeidentifyTemplate value.
    updateMask: Mask to control which fields get updated.
  """
    deidentifyTemplate = _messages.MessageField('GooglePrivacyDlpV2DeidentifyTemplate', 1)
    updateMask = _messages.StringField(2)