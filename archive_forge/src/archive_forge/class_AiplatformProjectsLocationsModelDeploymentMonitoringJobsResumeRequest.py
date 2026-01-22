from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelDeploymentMonitoringJobsResumeRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelDeploymentMonitoringJobsResumeRequest
  object.

  Fields:
    googleCloudAiplatformV1ResumeModelDeploymentMonitoringJobRequest: A
      GoogleCloudAiplatformV1ResumeModelDeploymentMonitoringJobRequest
      resource to be passed as the request body.
    name: Required. The resource name of the ModelDeploymentMonitoringJob to
      resume. Format: `projects/{project}/locations/{location}/modelDeployment
      MonitoringJobs/{model_deployment_monitoring_job}`
  """
    googleCloudAiplatformV1ResumeModelDeploymentMonitoringJobRequest = _messages.MessageField('GoogleCloudAiplatformV1ResumeModelDeploymentMonitoringJobRequest', 1)
    name = _messages.StringField(2, required=True)