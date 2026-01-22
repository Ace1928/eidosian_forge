from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsJobRunsGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsJobRunsGe
  tRequest object.

  Fields:
    name: Required. Name of the `JobRun`. Format must be `projects/{project_id
      }/locations/{location_name}/deliveryPipelines/{pipeline_name}/releases/{
      release_name}/rollouts/{rollout_name}/jobRuns/{job_run_name}`.
  """
    name = _messages.StringField(1, required=True)