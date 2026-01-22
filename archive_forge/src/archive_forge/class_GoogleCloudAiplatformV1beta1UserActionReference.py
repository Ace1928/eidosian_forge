from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1UserActionReference(_messages.Message):
    """References an API call. It contains more information about long running
  operation and Jobs that are triggered by the API call.

  Fields:
    dataLabelingJob: For API calls that start a LabelingJob. Resource name of
      the LabelingJob. Format: `projects/{project}/locations/{location}/dataLa
      belingJobs/{data_labeling_job}`
    method: The method name of the API RPC call. For example,
      "/google.cloud.aiplatform.{apiVersion}.DatasetService.CreateDataset"
    operation: For API calls that return a long running operation. Resource
      name of the long running operation. Format:
      `projects/{project}/locations/{location}/operations/{operation}`
  """
    dataLabelingJob = _messages.StringField(1)
    method = _messages.StringField(2)
    operation = _messages.StringField(3)