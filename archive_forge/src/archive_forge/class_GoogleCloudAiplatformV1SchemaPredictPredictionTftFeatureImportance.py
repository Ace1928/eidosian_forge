from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictPredictionTftFeatureImportance(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaPredictPredictionTftFeatureImportance
  object.

  Fields:
    attributeColumns: A string attribute.
    attributeWeights: A number attribute.
    contextColumns: A string attribute.
    contextWeights: TFT feature importance values. Each pair for
      {context/horizon/attribute} should have the same shape since the weight
      corresponds to the column names.
    horizonColumns: A string attribute.
    horizonWeights: A number attribute.
  """
    attributeColumns = _messages.StringField(1, repeated=True)
    attributeWeights = _messages.FloatField(2, repeated=True, variant=_messages.Variant.FLOAT)
    contextColumns = _messages.StringField(3, repeated=True)
    contextWeights = _messages.FloatField(4, repeated=True, variant=_messages.Variant.FLOAT)
    horizonColumns = _messages.StringField(5, repeated=True)
    horizonWeights = _messages.FloatField(6, repeated=True, variant=_messages.Variant.FLOAT)