from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomModuleValidationErrors(_messages.Message):
    """A list of zero or more errors encountered while validating the uploaded
  configuration of an Event Threat Detection Custom Module.

  Fields:
    errors: A CustomModuleValidationError attribute.
  """
    errors = _messages.MessageField('CustomModuleValidationError', 1, repeated=True)