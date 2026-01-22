from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringComponentConfig(_messages.Message):
    """MonitoringComponentConfig is cluster monitoring component configuration.

  Enums:
    EnableComponentsValueListEntryValuesEnum:

  Fields:
    enableComponents: Select components to collect metrics. An empty set would
      disable all monitoring.
  """

    class EnableComponentsValueListEntryValuesEnum(_messages.Enum):
        """EnableComponentsValueListEntryValuesEnum enum type.

    Values:
      COMPONENT_UNSPECIFIED: Default value. This shouldn't be used.
      SYSTEM_COMPONENTS: system components
      WORKLOADS: workloads
      APISERVER: kube-apiserver
      SCHEDULER: kube-scheduler
      CONTROLLER_MANAGER: kube-controller-manager
      STORAGE: Storage
      HPA: Horizontal Pod Autoscaling
      POD: Pod
      DAEMONSET: DaemonSet
      DEPLOYMENT: Deployment
      STATEFULSET: Statefulset
      CADVISOR: CADVISOR
      KUBELET: KUBELET
      DCGM: NVIDIA Data Center GPU Manager (DCGM)
    """
        COMPONENT_UNSPECIFIED = 0
        SYSTEM_COMPONENTS = 1
        WORKLOADS = 2
        APISERVER = 3
        SCHEDULER = 4
        CONTROLLER_MANAGER = 5
        STORAGE = 6
        HPA = 7
        POD = 8
        DAEMONSET = 9
        DEPLOYMENT = 10
        STATEFULSET = 11
        CADVISOR = 12
        KUBELET = 13
        DCGM = 14
    enableComponents = _messages.EnumField('EnableComponentsValueListEntryValuesEnum', 1, repeated=True)