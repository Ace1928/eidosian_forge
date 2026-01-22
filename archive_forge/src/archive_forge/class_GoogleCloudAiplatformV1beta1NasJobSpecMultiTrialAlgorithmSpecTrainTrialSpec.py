from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1NasJobSpecMultiTrialAlgorithmSpecTrainTrialSpec(_messages.Message):
    """Represent spec for train trials.

  Fields:
    frequency: Required. Frequency of search trials to start train stage. Top
      N [TrainTrialSpec.max_parallel_trial_count] search trials will be
      trained for every M [TrainTrialSpec.frequency] trials searched.
    maxParallelTrialCount: Required. The maximum number of trials to run in
      parallel.
    trainTrialJobSpec: Required. The spec of a train trial job. The same spec
      applies to all train trials.
  """
    frequency = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxParallelTrialCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    trainTrialJobSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1CustomJobSpec', 3)