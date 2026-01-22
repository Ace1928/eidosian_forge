from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PipelineCondition(_messages.Message):
    """PipelineCondition contains all conditions relevant to a Delivery
  Pipeline.

  Fields:
    pipelineReadyCondition: Details around the Pipeline's overall status.
    targetsPresentCondition: Details around targets enumerated in the
      pipeline.
    targetsTypeCondition: Details on the whether the targets enumerated in the
      pipeline are of the same type.
  """
    pipelineReadyCondition = _messages.MessageField('PipelineReadyCondition', 1)
    targetsPresentCondition = _messages.MessageField('TargetsPresentCondition', 2)
    targetsTypeCondition = _messages.MessageField('TargetsTypeCondition', 3)