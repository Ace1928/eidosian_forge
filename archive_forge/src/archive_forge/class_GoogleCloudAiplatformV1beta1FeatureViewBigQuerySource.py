from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureViewBigQuerySource(_messages.Message):
    """A GoogleCloudAiplatformV1beta1FeatureViewBigQuerySource object.

  Fields:
    entityIdColumns: Required. Columns to construct entity_id / row keys.
    uri: Required. The BigQuery view URI that will be materialized on each
      sync trigger based on FeatureView.SyncConfig.
  """
    entityIdColumns = _messages.StringField(1, repeated=True)
    uri = _messages.StringField(2)