from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectValueValuesEnum(_messages.Enum):
    """Required. The taint effect.

    Values:
      EFFECT_UNSPECIFIED: Not set.
      NO_SCHEDULE: Do not allow new pods to schedule onto the node unless they
        tolerate the taint, but allow all pods submitted to Kubelet without
        going through the scheduler to start, and allow all already-running
        pods to continue running. Enforced by the scheduler.
      PREFER_NO_SCHEDULE: Like TaintEffectNoSchedule, but the scheduler tries
        not to schedule new pods onto the node, rather than prohibiting new
        pods from scheduling onto the node entirely. Enforced by the
        scheduler.
      NO_EXECUTE: Evict any already-running pods that do not tolerate the
        taint. Currently enforced by NodeController.
    """
    EFFECT_UNSPECIFIED = 0
    NO_SCHEDULE = 1
    PREFER_NO_SCHEDULE = 2
    NO_EXECUTE = 3