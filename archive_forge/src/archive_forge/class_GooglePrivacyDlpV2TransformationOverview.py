from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransformationOverview(_messages.Message):
    """Overview of the modifications that occurred.

  Fields:
    transformationSummaries: Transformations applied to the dataset.
    transformedBytes: Total size in bytes that were transformed in some way.
  """
    transformationSummaries = _messages.MessageField('GooglePrivacyDlpV2TransformationSummary', 1, repeated=True)
    transformedBytes = _messages.IntegerField(2)