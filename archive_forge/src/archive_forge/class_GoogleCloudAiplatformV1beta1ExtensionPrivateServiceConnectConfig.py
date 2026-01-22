from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExtensionPrivateServiceConnectConfig(_messages.Message):
    """PrivateExtensionConfig configuration for the extension.

  Fields:
    serviceDirectory: Required. The Service Directory resource name in which
      the service endpoints associated to the extension are registered.
      Format: `projects/{project_id}/locations/{location_id}/namespaces/{names
      pace_id}/services/{service_id}` - The Vertex AI Extension Service Agent
      (https://cloud.google.com/vertex-ai/docs/general/access-control#service-
      agents) should be granted `servicedirectory.viewer` and
      `servicedirectory.pscAuthorizedService` roles on the resource.
  """
    serviceDirectory = _messages.StringField(1)