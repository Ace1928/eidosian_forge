from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreemptibilityValueValuesEnum(_messages.Enum):
    """Optional. Specifies the preemptibility of the instance group.The
    default value for master and worker groups is NON_PREEMPTIBLE. This
    default cannot be changed.The default value for secondary instances is
    PREEMPTIBLE.

    Values:
      PREEMPTIBILITY_UNSPECIFIED: Preemptibility is unspecified, the system
        will choose the appropriate setting for each instance group.
      NON_PREEMPTIBLE: Instances are non-preemptible.This option is allowed
        for all instance groups and is the only valid value for Master and
        Worker instance groups.
      PREEMPTIBLE: Instances are preemptible
        (https://cloud.google.com/compute/docs/instances/preemptible).This
        option is allowed only for secondary worker
        (https://cloud.google.com/dataproc/docs/concepts/compute/secondary-
        vms) groups.
      SPOT: Instances are Spot VMs
        (https://cloud.google.com/compute/docs/instances/spot).This option is
        allowed only for secondary worker
        (https://cloud.google.com/dataproc/docs/concepts/compute/secondary-
        vms) groups. Spot VMs are the latest version of preemptible VMs
        (https://cloud.google.com/compute/docs/instances/preemptible), and
        provide additional features.
    """
    PREEMPTIBILITY_UNSPECIFIED = 0
    NON_PREEMPTIBLE = 1
    PREEMPTIBLE = 2
    SPOT = 3