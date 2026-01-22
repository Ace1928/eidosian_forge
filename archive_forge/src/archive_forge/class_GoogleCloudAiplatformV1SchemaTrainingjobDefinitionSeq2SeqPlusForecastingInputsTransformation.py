from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsTransformation(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForecasti
  ngInputsTransformation object.

  Fields:
    auto: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForec
      astingInputsTransformationAutoTransformation attribute.
    categorical: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPl
      usForecastingInputsTransformationCategoricalTransformation attribute.
    numeric: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusFo
      recastingInputsTransformationNumericTransformation attribute.
    text: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForec
      astingInputsTransformationTextTransformation attribute.
    timestamp: A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlus
      ForecastingInputsTransformationTimestampTransformation attribute.
  """
    auto = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsTransformationAutoTransformation', 1)
    categorical = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsTransformationCategoricalTransformation', 2)
    numeric = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsTransformationNumericTransformation', 3)
    text = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsTransformationTextTransformation', 4)
    timestamp = _messages.MessageField('GoogleCloudAiplatformV1SchemaTrainingjobDefinitionSeq2SeqPlusForecastingInputsTransformationTimestampTransformation', 5)