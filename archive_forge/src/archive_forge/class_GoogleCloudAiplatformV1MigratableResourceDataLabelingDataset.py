from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1MigratableResourceDataLabelingDataset(_messages.Message):
    """Represents one Dataset in datalabeling.googleapis.com.

  Fields:
    dataLabelingAnnotatedDatasets: The migratable AnnotatedDataset in
      datalabeling.googleapis.com belongs to the data labeling Dataset.
    dataset: Full resource name of data labeling Dataset. Format:
      `projects/{project}/datasets/{dataset}`.
    datasetDisplayName: The Dataset's display name in
      datalabeling.googleapis.com.
  """
    dataLabelingAnnotatedDatasets = _messages.MessageField('GoogleCloudAiplatformV1MigratableResourceDataLabelingDatasetDataLabelingAnnotatedDataset', 1, repeated=True)
    dataset = _messages.StringField(2)
    datasetDisplayName = _messages.StringField(3)