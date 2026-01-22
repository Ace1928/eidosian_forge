from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1GrpcOperationConfig(_messages.Message):
    """Binds the resources in a proxy or remote service with the gRPC operation
  and its associated quota enforcement.

  Fields:
    apiSource: Required. Name of the API proxy with which the gRPC operation
      and quota are associated.
    attributes: Custom attributes associated with the operation.
    methods: List of unqualified gRPC method names for the proxy to which
      quota will be applied. If this field is empty, the Quota will apply to
      all operations on the gRPC service defined on the proxy. Example: Given
      a proxy that is configured to serve com.petstore.PetService, the methods
      com.petstore.PetService.ListPets and com.petstore.PetService.GetPet
      would be specified here as simply ["ListPets", "GetPet"].
    quota: Quota parameters to be enforced for the methods and API source
      combination. If none are specified, quota enforcement will not be done.
    service: Required. gRPC Service name associated to be associated with the
      API proxy, on which quota rules can be applied upon.
  """
    apiSource = _messages.StringField(1)
    attributes = _messages.MessageField('GoogleCloudApigeeV1Attribute', 2, repeated=True)
    methods = _messages.StringField(3, repeated=True)
    quota = _messages.MessageField('GoogleCloudApigeeV1Quota', 4)
    service = _messages.StringField(5)