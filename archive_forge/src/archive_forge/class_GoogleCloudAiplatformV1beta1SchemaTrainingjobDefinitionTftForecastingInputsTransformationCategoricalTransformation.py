from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionTftForecastingInputsTransformationCategoricalTransformation(_messages.Message):
    """Training pipeline will perform following transformation functions. * The
  categorical string as is--no change to case, punctuation, spelling, tense,
  and so on. * Convert the category name to a dictionary lookup index and
  generate an embedding for each index. * Categories that appear less than 5
  times in the training dataset are treated as the "unknown" category. The
  "unknown" category gets its own special lookup index and resulting
  embedding.

  Fields:
    columnName: A string attribute.
  """
    columnName = _messages.StringField(1)