from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlForecastingInputsTransformationAutoTransformation(_messages.Message):
    """Training pipeline will infer the proper transformation based on the
  statistic of dataset.

  Fields:
    columnName: A string attribute.
  """
    columnName = _messages.StringField(1)