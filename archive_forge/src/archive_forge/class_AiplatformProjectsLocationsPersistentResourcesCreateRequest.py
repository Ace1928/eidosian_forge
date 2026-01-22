from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPersistentResourcesCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsPersistentResourcesCreateRequest object.

  Fields:
    googleCloudAiplatformV1PersistentResource: A
      GoogleCloudAiplatformV1PersistentResource resource to be passed as the
      request body.
    parent: Required. The resource name of the Location to create the
      PersistentResource in. Format: `projects/{project}/locations/{location}`
    persistentResourceId: Required. The ID to use for the PersistentResource,
      which become the final component of the PersistentResource's resource
      name. The maximum length is 63 characters, and valid characters are
      `/^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$/`.
  """
    googleCloudAiplatformV1PersistentResource = _messages.MessageField('GoogleCloudAiplatformV1PersistentResource', 1)
    parent = _messages.StringField(2, required=True)
    persistentResourceId = _messages.StringField(3)