from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelsExportRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelsExportRequest object.

  Fields:
    googleCloudAiplatformV1ExportModelRequest: A
      GoogleCloudAiplatformV1ExportModelRequest resource to be passed as the
      request body.
    name: Required. The resource name of the Model to export. The resource
      name may contain version id or version alias to specify the version, if
      no version is specified, the default version will be exported.
  """
    googleCloudAiplatformV1ExportModelRequest = _messages.MessageField('GoogleCloudAiplatformV1ExportModelRequest', 1)
    name = _messages.StringField(2, required=True)