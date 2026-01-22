from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsIndexesPatchRequest(_messages.Message):
    """A AiplatformProjectsLocationsIndexesPatchRequest object.

  Fields:
    googleCloudAiplatformV1Index: A GoogleCloudAiplatformV1Index resource to
      be passed as the request body.
    name: Output only. The resource name of the Index.
    updateMask: The update mask applies to the resource. For the `FieldMask`
      definition, see google.protobuf.FieldMask.
  """
    googleCloudAiplatformV1Index = _messages.MessageField('GoogleCloudAiplatformV1Index', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)