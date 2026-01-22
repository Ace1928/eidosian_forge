from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateEventThreatDetectionCustomModuleRequest(_messages.Message):
    """Request to validate an Event Threat Detection custom module.

  Fields:
    rawText: Required. The raw text of the module's contents. Used to generate
      error messages.
    type: Required. The type of the module (e.g. CONFIGURABLE_BAD_IP).
  """
    rawText = _messages.StringField(1)
    type = _messages.StringField(2)