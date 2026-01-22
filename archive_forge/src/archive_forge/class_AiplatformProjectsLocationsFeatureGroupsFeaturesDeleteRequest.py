from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeatureGroupsFeaturesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeatureGroupsFeaturesDeleteRequest object.

  Fields:
    name: Required. The name of the Features to be deleted. Format: `projects/
      {project}/locations/{location}/featurestores/{featurestore}/entityTypes/
      {entity_type}/features/{feature}` `projects/{project}/locations/{locatio
      n}/featureGroups/{feature_group}/features/{feature}`
  """
    name = _messages.StringField(1, required=True)