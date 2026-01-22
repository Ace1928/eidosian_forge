from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BinaryauthorizationProjectsAttestorsCreateRequest(_messages.Message):
    """A BinaryauthorizationProjectsAttestorsCreateRequest object.

  Fields:
    attestor: A Attestor resource to be passed as the request body.
    attestorId: Required. The attestors ID.
    parent: Required. The parent of this attestor.
  """
    attestor = _messages.MessageField('Attestor', 1)
    attestorId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)