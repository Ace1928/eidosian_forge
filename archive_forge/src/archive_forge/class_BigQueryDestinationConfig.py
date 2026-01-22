from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigQueryDestinationConfig(_messages.Message):
    """BigQuery destination configuration

  Fields:
    appendOnly: Append only mode
    dataFreshness: The guaranteed data freshness (in seconds) when querying
      tables created by the stream. Editing this field will only affect new
      tables created in the future, but existing tables will not be impacted.
      Lower values mean that queries will return fresher data, but may result
      in higher cost.
    merge: The standard mode
    singleTargetDataset: Single destination dataset.
    sourceHierarchyDatasets: Source hierarchy datasets.
  """
    appendOnly = _messages.MessageField('AppendOnly', 1)
    dataFreshness = _messages.StringField(2)
    merge = _messages.MessageField('Merge', 3)
    singleTargetDataset = _messages.MessageField('SingleTargetDataset', 4)
    sourceHierarchyDatasets = _messages.MessageField('SourceHierarchyDatasets', 5)