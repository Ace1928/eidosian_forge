from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1FeatureViewFeatureRegistrySource(_messages.Message):
    """A Feature Registry source for features that need to be synced to Online
  Store.

  Fields:
    featureGroups: Required. List of features that need to be synced to Online
      Store.
    projectNumber: Optional. The project number of the parent project of the
      Feature Groups.
  """
    featureGroups = _messages.MessageField('GoogleCloudAiplatformV1FeatureViewFeatureRegistrySourceFeatureGroup', 1, repeated=True)
    projectNumber = _messages.IntegerField(2)