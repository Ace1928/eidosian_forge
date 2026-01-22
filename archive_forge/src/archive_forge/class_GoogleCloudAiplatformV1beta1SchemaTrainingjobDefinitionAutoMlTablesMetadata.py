from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesMetadata(_messages.Message):
    """Model metadata specific to AutoML Tables.

  Fields:
    evaluatedDataItemsBigqueryUri: BigQuery destination uri for exported
      evaluated examples.
    trainCostMilliNodeHours: Output only. The actual training cost of the
      model, expressed in milli node hours, i.e. 1,000 value in this field
      means 1 node hour. Guaranteed to not exceed the train budget.
  """
    evaluatedDataItemsBigqueryUri = _messages.StringField(1)
    trainCostMilliNodeHours = _messages.IntegerField(2)