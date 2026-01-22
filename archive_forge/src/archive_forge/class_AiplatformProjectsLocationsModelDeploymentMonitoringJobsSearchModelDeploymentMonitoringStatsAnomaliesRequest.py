from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsModelDeploymentMonitoringJobsSearchModelDeploymentMonitoringStatsAnomaliesRequest(_messages.Message):
    """A AiplatformProjectsLocationsModelDeploymentMonitoringJobsSearchModelDep
  loymentMonitoringStatsAnomaliesRequest object.

  Fields:
    googleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomaliesReques
      t: A GoogleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomalie
      sRequest resource to be passed as the request body.
    modelDeploymentMonitoringJob: Required. ModelDeploymentMonitoring Job
      resource name. Format: `projects/{project}/locations/{location}/modelDep
      loymentMonitoringJobs/{model_deployment_monitoring_job}`
  """
    googleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomaliesRequest = _messages.MessageField('GoogleCloudAiplatformV1SearchModelDeploymentMonitoringStatsAnomaliesRequest', 1)
    modelDeploymentMonitoringJob = _messages.StringField(2, required=True)