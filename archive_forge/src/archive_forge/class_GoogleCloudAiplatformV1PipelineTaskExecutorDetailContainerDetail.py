from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PipelineTaskExecutorDetailContainerDetail(_messages.Message):
    """The detail of a container execution. It contains the job names of the
  lifecycle of a container execution.

  Fields:
    failedMainJobs: Output only. The names of the previously failed CustomJob
      for the main container executions. The list includes the all attempts in
      chronological order.
    failedPreCachingCheckJobs: Output only. The names of the previously failed
      CustomJob for the pre-caching-check container executions. This job will
      be available if the PipelineJob.pipeline_spec specifies the
      `pre_caching_check` hook in the lifecycle events. The list includes the
      all attempts in chronological order.
    mainJob: Output only. The name of the CustomJob for the main container
      execution.
    preCachingCheckJob: Output only. The name of the CustomJob for the pre-
      caching-check container execution. This job will be available if the
      PipelineJob.pipeline_spec specifies the `pre_caching_check` hook in the
      lifecycle events.
  """
    failedMainJobs = _messages.StringField(1, repeated=True)
    failedPreCachingCheckJobs = _messages.StringField(2, repeated=True)
    mainJob = _messages.StringField(3)
    preCachingCheckJob = _messages.StringField(4)