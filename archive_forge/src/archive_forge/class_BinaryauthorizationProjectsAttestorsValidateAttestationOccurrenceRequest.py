from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BinaryauthorizationProjectsAttestorsValidateAttestationOccurrenceRequest(_messages.Message):
    """A
  BinaryauthorizationProjectsAttestorsValidateAttestationOccurrenceRequest
  object.

  Fields:
    attestor: Required. The resource name of the Attestor of the occurrence,
      in the format `projects/*/attestors/*`.
    validateAttestationOccurrenceRequest: A
      ValidateAttestationOccurrenceRequest resource to be passed as the
      request body.
  """
    attestor = _messages.StringField(1, required=True)
    validateAttestationOccurrenceRequest = _messages.MessageField('ValidateAttestationOccurrenceRequest', 2)