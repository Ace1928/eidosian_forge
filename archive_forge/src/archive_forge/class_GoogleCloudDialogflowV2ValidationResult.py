from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ValidationResult(_messages.Message):
    """Represents the output of agent validation.

  Fields:
    validationErrors: Contains all validation errors.
  """
    validationErrors = _messages.MessageField('GoogleCloudDialogflowV2ValidationError', 1, repeated=True)