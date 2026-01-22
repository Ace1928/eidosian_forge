from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ImportFeatureValuesResponse(_messages.Message):
    """Response message for FeaturestoreService.ImportFeatureValues.

  Fields:
    importedEntityCount: Number of entities that have been imported by the
      operation.
    importedFeatureValueCount: Number of Feature values that have been
      imported by the operation.
    invalidRowCount: The number of rows in input source that weren't imported
      due to either * Not having any featureValues. * Having a null entityId.
      * Having a null timestamp. * Not being parsable (applicable for CSV
      sources).
    timestampOutsideRetentionRowsCount: The number rows that weren't ingested
      due to having feature timestamps outside the retention boundary.
  """
    importedEntityCount = _messages.IntegerField(1)
    importedFeatureValueCount = _messages.IntegerField(2)
    invalidRowCount = _messages.IntegerField(3)
    timestampOutsideRetentionRowsCount = _messages.IntegerField(4)