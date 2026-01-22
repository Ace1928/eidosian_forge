from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesPatchRequest(_messages.Message):
    """A RunProjectsLocationsServicesPatchRequest object.

  Fields:
    allowMissing: Optional. If set to true, and if the Service does not exist,
      it will create a new one. The caller must have 'run.services.create'
      permissions if this is set to true and the Service does not exist.
    googleCloudRunV2Service: A GoogleCloudRunV2Service resource to be passed
      as the request body.
    name: The fully qualified name of this Service. In CreateServiceRequest,
      this field is ignored, and instead composed from
      CreateServiceRequest.parent and CreateServiceRequest.service_id. Format:
      projects/{project}/locations/{location}/services/{service_id}
    updateMask: Optional. The list of fields to be updated.
    validateOnly: Indicates that the request should be validated and default
      values populated, without persisting the request or updating any
      resources.
  """
    allowMissing = _messages.BooleanField(1)
    googleCloudRunV2Service = _messages.MessageField('GoogleCloudRunV2Service', 2)
    name = _messages.StringField(3, required=True)
    updateMask = _messages.StringField(4)
    validateOnly = _messages.BooleanField(5)