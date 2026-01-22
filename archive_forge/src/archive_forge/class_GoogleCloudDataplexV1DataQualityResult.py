from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataQualityResult(_messages.Message):
    """The output of a DataQualityScan.

  Fields:
    columns: Output only. A list of results at the column level.A column will
      have a corresponding DataQualityColumnResult if and only if there is at
      least one rule with the 'column' field set to it.
    dimensions: A list of results at the dimension level.A dimension will have
      a corresponding DataQualityDimensionResult if and only if there is at
      least one rule with the 'dimension' field set to it.
    passed: Overall data quality result -- true if all rules passed.
    postScanActionsResult: Output only. The result of post scan actions.
    rowCount: The count of rows processed.
    rules: A list of all the rules in a job, and their results.
    scannedData: The data scanned for this result.
    score: Output only. The overall data quality score.The score ranges
      between 0, 100 (up to two decimal points).
  """
    columns = _messages.MessageField('GoogleCloudDataplexV1DataQualityColumnResult', 1, repeated=True)
    dimensions = _messages.MessageField('GoogleCloudDataplexV1DataQualityDimensionResult', 2, repeated=True)
    passed = _messages.BooleanField(3)
    postScanActionsResult = _messages.MessageField('GoogleCloudDataplexV1DataQualityResultPostScanActionsResult', 4)
    rowCount = _messages.IntegerField(5)
    rules = _messages.MessageField('GoogleCloudDataplexV1DataQualityRuleResult', 6, repeated=True)
    scannedData = _messages.MessageField('GoogleCloudDataplexV1ScannedData', 7)
    score = _messages.FloatField(8, variant=_messages.Variant.FLOAT)