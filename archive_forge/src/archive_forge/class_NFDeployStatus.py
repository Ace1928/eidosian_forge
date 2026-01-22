from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NFDeployStatus(_messages.Message):
    """Deployment status of NFDeploy.

  Fields:
    readyNfs: Output only. Total number of NFs targeted by this deployment
      with a Ready Condition set.
    sites: Output only. Per-Site Status.
    targetedNfs: Output only. Total number of NFs targeted by this deployment
  """
    readyNfs = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    sites = _messages.MessageField('NFDeploySiteStatus', 2, repeated=True)
    targetedNfs = _messages.IntegerField(3, variant=_messages.Variant.INT32)