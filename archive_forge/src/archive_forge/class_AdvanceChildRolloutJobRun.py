from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdvanceChildRolloutJobRun(_messages.Message):
    """AdvanceChildRolloutJobRun contains information specific to a
  advanceChildRollout `JobRun`.

  Fields:
    rollout: Output only. Name of the `ChildRollout`. Format is `projects/{pro
      ject}/locations/{location}/deliveryPipelines/{deliveryPipeline}/releases
      /{release}/rollouts/a-z{0,62}`.
    rolloutPhaseId: Output only. the ID of the ChildRollout's Phase.
  """
    rollout = _messages.StringField(1)
    rolloutPhaseId = _messages.StringField(2)