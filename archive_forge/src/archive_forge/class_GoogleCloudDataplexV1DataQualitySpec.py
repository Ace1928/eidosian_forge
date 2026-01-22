from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualitySpec(_messages.Message):
    """DataQualityScan related setting.

  Fields:
    postScanActions: Optional. Actions to take upon job completion.
    rowFilter: Optional. A filter applied to all rows in a single DataScan
      job. The filter needs to be a valid SQL expression for a WHERE clause in
      BigQuery standard SQL syntax. Example: col1 >= 0 AND col2 < 10
    rules: Required. The list of rules to evaluate against a data source. At
      least one rule is required.
    samplingPercent: Optional. The percentage of the records to be selected
      from the dataset for DataScan. Value can range between 0.0 and 100.0
      with up to 3 significant decimal digits. Sampling is not applied if
      sampling_percent is not specified, 0 or 100.
  """
    postScanActions = _messages.MessageField('GoogleCloudDataplexV1DataQualitySpecPostScanActions', 1)
    rowFilter = _messages.StringField(2)
    rules = _messages.MessageField('GoogleCloudDataplexV1DataQualityRule', 3, repeated=True)
    samplingPercent = _messages.FloatField(4, variant=_messages.Variant.FLOAT)