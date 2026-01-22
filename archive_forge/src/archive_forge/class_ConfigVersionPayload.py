from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigVersionPayload(_messages.Message):
    """Message for storing a ConfigVersion resource's payload data based upon
  its type.

  Fields:
    rawPayload: Optional. REQUIRED for a ConfigType of RAW.
    templateValuesPayload: Optional. REQUIRED for a ConfigType of TEMPLATED.
  """
    rawPayload = _messages.MessageField('RawPayload', 1)
    templateValuesPayload = _messages.MessageField('TemplateValuesPayload', 2)