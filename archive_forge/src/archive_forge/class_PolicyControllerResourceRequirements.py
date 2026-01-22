from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyControllerResourceRequirements(_messages.Message):
    """ResourceRequirements describes the compute resource requirements.

  Fields:
    limits: Limits describes the maximum amount of compute resources allowed
      for use by the running container.
    requests: Requests describes the amount of compute resources reserved for
      the container by the kube-scheduler.
  """
    limits = _messages.MessageField('PolicyControllerResourceList', 1)
    requests = _messages.MessageField('PolicyControllerResourceList', 2)