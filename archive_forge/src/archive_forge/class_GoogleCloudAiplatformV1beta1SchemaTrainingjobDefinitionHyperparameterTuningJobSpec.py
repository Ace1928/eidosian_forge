from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionHyperparameterTuningJobSpec(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionHyperparameterT
  uningJobSpec object.

  Fields:
    maxFailedTrialCount: The number of failed Trials that need to be seen
      before failing the HyperparameterTuningJob. If set to 0, Vertex AI
      decides how many Trials must fail before the whole job fails.
    maxTrialCount: The desired total number of Trials.
    parallelTrialCount: The desired number of Trials to run in parallel.
    studySpec: Study configuration of the HyperparameterTuningJob.
    trialJobSpec: The spec of a trial job. The same spec applies to the
      CustomJobs created in all the trials.
  """
    maxFailedTrialCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxTrialCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    parallelTrialCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    studySpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpec', 4)
    trialJobSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1CustomJobSpec', 5)