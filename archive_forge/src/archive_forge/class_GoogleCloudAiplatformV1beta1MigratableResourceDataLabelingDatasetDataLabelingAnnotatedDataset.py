from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1MigratableResourceDataLabelingDatasetDataLabelingAnnotatedDataset(_messages.Message):
    """Represents one AnnotatedDataset in datalabeling.googleapis.com.

  Fields:
    annotatedDataset: Full resource name of data labeling AnnotatedDataset.
      Format: `projects/{project}/datasets/{dataset}/annotatedDatasets/{annota
      ted_dataset}`.
    annotatedDatasetDisplayName: The AnnotatedDataset's display name in
      datalabeling.googleapis.com.
  """
    annotatedDataset = _messages.StringField(1)
    annotatedDatasetDisplayName = _messages.StringField(2)