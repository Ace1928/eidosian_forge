from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsAspectTypesPatchRequest(_messages.Message):
    """A DataplexProjectsLocationsAspectTypesPatchRequest object.

  Fields:
    googleCloudDataplexV1AspectType: A GoogleCloudDataplexV1AspectType
      resource to be passed as the request body.
    name: Output only. The relative resource name of the AspectType, of the
      form: projects/{project_number}/locations/{location_id}/aspectTypes/{asp
      ect_type_id}.
    updateMask: Required. Mask of fields to update.
    validateOnly: Optional. Only validate the request, but do not perform
      mutations. The default is false.
  """
    googleCloudDataplexV1AspectType = _messages.MessageField('GoogleCloudDataplexV1AspectType', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)