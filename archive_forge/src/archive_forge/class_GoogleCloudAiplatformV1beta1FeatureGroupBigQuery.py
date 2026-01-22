from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureGroupBigQuery(_messages.Message):
    """Input source type for BigQuery Tables and Views.

  Fields:
    bigQuerySource: Required. Immutable. The BigQuery source URI that points
      to either a BigQuery Table or View.
    entityIdColumns: Optional. Columns to construct entity_id / row keys. If
      not provided defaults to `entity_id`.
  """
    bigQuerySource = _messages.MessageField('GoogleCloudAiplatformV1beta1BigQuerySource', 1)
    entityIdColumns = _messages.StringField(2, repeated=True)