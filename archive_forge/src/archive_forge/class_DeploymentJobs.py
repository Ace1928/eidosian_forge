from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeploymentJobs(_messages.Message):
    """Deployment job composition.

  Fields:
    deployJob: Output only. The deploy Job. This is the deploy job in the
      phase.
    postdeployJob: Output only. The postdeploy Job, which is the last job on
      the phase.
    predeployJob: Output only. The predeploy Job, which is the first job on
      the phase.
    verifyJob: Output only. The verify Job. Runs after a deploy if the deploy
      succeeds.
  """
    deployJob = _messages.MessageField('Job', 1)
    postdeployJob = _messages.MessageField('Job', 2)
    predeployJob = _messages.MessageField('Job', 3)
    verifyJob = _messages.MessageField('Job', 4)