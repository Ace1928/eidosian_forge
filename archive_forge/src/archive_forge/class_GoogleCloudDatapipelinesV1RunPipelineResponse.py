from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatapipelinesV1RunPipelineResponse(_messages.Message):
    """Response message for RunPipeline

  Fields:
    job: Job that was created as part of RunPipeline operation.
  """
    job = _messages.MessageField('GoogleCloudDatapipelinesV1Job', 1)