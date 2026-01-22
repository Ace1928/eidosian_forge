from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PipelineTaskExecutorDetail(_messages.Message):
    """The runtime detail of a pipeline executor.

  Fields:
    containerDetail: Output only. The detailed info for a container executor.
    customJobDetail: Output only. The detailed info for a custom job executor.
  """
    containerDetail = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineTaskExecutorDetailContainerDetail', 1)
    customJobDetail = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineTaskExecutorDetailCustomJobDetail', 2)