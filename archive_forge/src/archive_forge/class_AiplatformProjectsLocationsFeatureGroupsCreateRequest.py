from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureGroupsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureGroupsCreateRequest object.

  Fields:
    featureGroupId: Required. The ID to use for this FeatureGroup, which will
      become the final component of the FeatureGroup's resource name. This
      value may be up to 60 characters, and valid characters are `[a-z0-9_]`.
      The first character cannot be a number. The value must be unique within
      the project and location.
    googleCloudAiplatformV1FeatureGroup: A GoogleCloudAiplatformV1FeatureGroup
      resource to be passed as the request body.
    parent: Required. The resource name of the Location to create
      FeatureGroups. Format: `projects/{project}/locations/{location}'`
  """
    featureGroupId = _messages.StringField(1)
    googleCloudAiplatformV1FeatureGroup = _messages.MessageField('GoogleCloudAiplatformV1FeatureGroup', 2)
    parent = _messages.StringField(3, required=True)