from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlImageSegmentationMetadata(_messages.Message):
    """A GoogleCloudAiplatformV1SchemaTrainingjobDefinitionAutoMlImageSegmentat
  ionMetadata object.

  Enums:
    SuccessfulStopReasonValueValuesEnum: For successful job completions, this
      is the reason why the job has finished.

  Fields:
    costMilliNodeHours: The actual training cost of creating this model,
      expressed in milli node hours, i.e. 1,000 value in this field means 1
      node hour. Guaranteed to not exceed inputs.budgetMilliNodeHours.
    successfulStopReason: For successful job completions, this is the reason
      why the job has finished.
  """

    class SuccessfulStopReasonValueValuesEnum(_messages.Enum):
        """For successful job completions, this is the reason why the job has
    finished.

    Values:
      SUCCESSFUL_STOP_REASON_UNSPECIFIED: Should not be set.
      BUDGET_REACHED: The inputs.budgetMilliNodeHours had been reached.
      MODEL_CONVERGED: Further training of the Model ceased to increase its
        quality, since it already has converged.
    """
        SUCCESSFUL_STOP_REASON_UNSPECIFIED = 0
        BUDGET_REACHED = 1
        MODEL_CONVERGED = 2
    costMilliNodeHours = _messages.IntegerField(1)
    successfulStopReason = _messages.EnumField('SuccessfulStopReasonValueValuesEnum', 2)