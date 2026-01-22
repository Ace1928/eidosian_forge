from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModelFeature(_messages.Message):
    """Representes a singular feature of a model. If the feature is
  `recognizer`, the release_state of the feature represents the release_state
  of the model

  Fields:
    feature: The name of the feature (Note: the feature can be `recognizer`)
    releaseState: The release state of the feature
  """
    feature = _messages.StringField(1)
    releaseState = _messages.StringField(2)