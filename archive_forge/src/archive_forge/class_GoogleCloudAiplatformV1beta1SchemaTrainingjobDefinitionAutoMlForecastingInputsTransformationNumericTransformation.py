from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationNumericTransformation(_messages.Message):
    """Training pipeline will perform following transformation functions. * The
  value converted to float32. * The z_score of the value. * log(value+1) when
  the value is greater than or equal to 0. Otherwise, this transformation is
  not applied and the value is considered a missing value. * z_score of
  log(value+1) when the value is greater than or equal to 0. Otherwise, this
  transformation is not applied and the value is considered a missing value. *
  A boolean value that indicates whether the value is valid.

  Fields:
    columnName: A string attribute.
  """
    columnName = _messages.StringField(1)