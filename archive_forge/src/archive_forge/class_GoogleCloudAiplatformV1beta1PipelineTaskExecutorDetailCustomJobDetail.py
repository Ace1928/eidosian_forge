from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PipelineTaskExecutorDetailCustomJobDetail(_messages.Message):
    """The detailed info for a custom job executor.

  Fields:
    failedJobs: Output only. The names of the previously failed CustomJob. The
      list includes the all attempts in chronological order.
    job: Output only. The name of the CustomJob.
  """
    failedJobs = _messages.StringField(1, repeated=True)
    job = _messages.StringField(2)