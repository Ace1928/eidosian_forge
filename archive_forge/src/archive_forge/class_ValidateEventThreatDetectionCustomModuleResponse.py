from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ValidateEventThreatDetectionCustomModuleResponse(_messages.Message):
    """Response to validating an Event Threat Detection custom module.

  Fields:
    errors: A list of errors returned by the validator. If the list is empty,
      there were no errors.
  """
    errors = _messages.MessageField('CustomModuleValidationErrors', 1)