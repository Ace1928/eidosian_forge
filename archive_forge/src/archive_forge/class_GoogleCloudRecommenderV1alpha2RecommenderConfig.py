from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2RecommenderConfig(_messages.Message):
    """Configuration for a Recommender.

  Fields:
    etag: Fingerprint of the RecommenderConfig. Provides optimistic locking
      when updating.
    name: Name of recommender config. Eg, projects/[PROJECT_NUMBER]/locations/
      [LOCATION]/recommenders/[RECOMMENDER_ID]/config
    recommenderGenerationConfig: RecommenderGenerationConfig which configures
      the Generation of recommendations for this recommender.
    revisionId: Output only. Immutable. The revision ID of the config. A new
      revision is committed whenever the config is changed in any way. The
      format is an 8-character hexadecimal string.
    updateTime: Last time when the config was updated.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2)
    recommenderGenerationConfig = _messages.MessageField('GoogleCloudRecommenderV1alpha2RecommenderGenerationConfig', 3)
    revisionId = _messages.StringField(4)
    updateTime = _messages.StringField(5)