from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListDatasetsResponse(_messages.Message):
    """Response message for DatasetService.ListDatasets.

  Fields:
    datasets: A list of Datasets that matches the specified filter in the
      request.
    nextPageToken: The standard List next-page token.
  """
    datasets = _messages.MessageField('GoogleCloudAiplatformV1Dataset', 1, repeated=True)
    nextPageToken = _messages.StringField(2)