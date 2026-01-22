from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootScoredToken(_messages.Message):
    """A token with its own score.

  Fields:
    endTokenScore: Each end_token_score is a logprob for how well the
      completion would end at a particular token. See http://google3/labs/lang
      uage/aida/config/proto/model_config.proto;l=376;rcl=573039459
    score: Each score is the logprob for the token in model response.
    token: A string attribute.
  """
    endTokenScore = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    score = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    token = _messages.StringField(3)