from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1GenerateDataQualityRulesResponse(_messages.Message):
    """Generate recommended DataQualityRules response.

  Fields:
    rule: Generated recommended {@link DataQualityRule}s.
  """
    rule = _messages.MessageField('GoogleCloudDataplexV1DataQualityRule', 1, repeated=True)