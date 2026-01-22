from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommenderProjectsLocationsRecommendersGetConfigRequest(_messages.Message):
    """A RecommenderProjectsLocationsRecommendersGetConfigRequest object.

  Fields:
    name: Required. Name of the Recommendation Config to get. Acceptable
      formats: * `projects/[PROJECT_NUMBER]/locations/global/recommenders/[REC
      OMMENDER_ID]/config` * `projects/[PROJECT_ID]/locations/global/recommend
      ers/[RECOMMENDER_ID]/config`
  """
    name = _messages.StringField(1, required=True)