from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OperationConfig(_messages.Message):
    """Binds the resources in an API proxy or remote service with the allowed
  REST methods and associated quota enforcement.

  Fields:
    apiSource: Required. Name of the API proxy or remote service with which
      the resources, methods, and quota are associated.
    attributes: Custom attributes associated with the operation.
    operations: List of resource/method pairs for the API proxy or remote
      service to which quota will applied. **Note**: Currently, you can
      specify only a single resource/method pair. The call will fail if more
      than one resource/method pair is provided.
    quota: Quota parameters to be enforced for the resources, methods, and API
      source combination. If none are specified, quota enforcement will not be
      done.
  """
    apiSource = _messages.StringField(1)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 2, repeated=True)
    operations = _messages.MessageField('GoogleCloudApigeeV1Operation', 3, repeated=True)
    quota = _messages.MessageField('GoogleCloudApigeeV1Quota', 4)