from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsKmsConfigsVerifyRequest(_messages.Message):
    """A NetappProjectsLocationsKmsConfigsVerifyRequest object.

  Fields:
    name: Required. Name of the KMS Config to be verified.
    verifyKmsConfigRequest: A VerifyKmsConfigRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    verifyKmsConfigRequest = _messages.MessageField('VerifyKmsConfigRequest', 2)