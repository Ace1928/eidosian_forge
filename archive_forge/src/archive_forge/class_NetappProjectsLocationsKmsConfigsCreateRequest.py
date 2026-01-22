from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsKmsConfigsCreateRequest(_messages.Message):
    """A NetappProjectsLocationsKmsConfigsCreateRequest object.

  Fields:
    kmsConfig: A KmsConfig resource to be passed as the request body.
    kmsConfigId: Required. Id of the requesting KmsConfig If auto-generating
      Id server-side, remove this field and id from the method_signature of
      Create RPC
    parent: Required. Value for parent.
  """
    kmsConfig = _messages.MessageField('KmsConfig', 1)
    kmsConfigId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)