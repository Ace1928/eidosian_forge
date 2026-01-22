from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesPatchRequest(_messages.Message):
    """A
  AiplatformProjectsLocationsFeaturestoresEntityTypesFeaturesPatchRequest
  object.

  Fields:
    googleCloudAiplatformV1Feature: A GoogleCloudAiplatformV1Feature resource
      to be passed as the request body.
    name: Immutable. Name of the Feature. Format: `projects/{project}/location
      s/{location}/featurestores/{featurestore}/entityTypes/{entity_type}/feat
      ures/{feature}` `projects/{project}/locations/{location}/featureGroups/{
      feature_group}/features/{feature}` The last part feature is assigned by
      the client. The feature can be up to 64 characters long and can consist
      only of ASCII Latin letters A-Z and a-z, underscore(_), and ASCII digits
      0-9 starting with a letter. The value will be unique given an entity
      type.
    updateMask: Field mask is used to specify the fields to be overwritten in
      the Features resource by the update. The fields specified in the
      update_mask are relative to the resource, not the full request. A field
      will be overwritten if it is in the mask. If the user does not provide a
      mask then only the non-empty fields present in the request will be
      overwritten. Set the update_mask to `*` to override all fields.
      Updatable fields: * `description` * `labels` * `disable_monitoring` (Not
      supported for FeatureRegistry Feature)
  """
    googleCloudAiplatformV1Feature = _messages.MessageField('GoogleCloudAiplatformV1Feature', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)