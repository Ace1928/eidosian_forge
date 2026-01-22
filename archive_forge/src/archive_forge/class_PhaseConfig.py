from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PhaseConfig(_messages.Message):
    """PhaseConfig represents the configuration for a phase in the custom
  canary deployment.

  Fields:
    percentage: Required. Percentage deployment for the phase.
    phaseId: Required. The ID to assign to the `Rollout` phase. This value
      must consist of lower-case letters, numbers, and hyphens, start with a
      letter and end with a letter or a number, and have a max length of 63
      characters. In other words, it must match the following regex:
      `^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$`.
    postdeploy: Optional. Configuration for the postdeploy job of this phase.
      If this is not configured, there will be no postdeploy job for this
      phase.
    predeploy: Optional. Configuration for the predeploy job of this phase. If
      this is not configured, there will be no predeploy job for this phase.
    profiles: Skaffold profiles to use when rendering the manifest for this
      phase. These are in addition to the profiles list specified in the
      `DeliveryPipeline` stage.
    verify: Whether to run verify tests after the deployment.
  """
    percentage = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    phaseId = _messages.StringField(2)
    postdeploy = _messages.MessageField('Postdeploy', 3)
    predeploy = _messages.MessageField('Predeploy', 4)
    profiles = _messages.StringField(5, repeated=True)
    verify = _messages.BooleanField(6)