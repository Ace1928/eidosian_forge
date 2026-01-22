from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingComponentConfig(_messages.Message):
    """LoggingComponentConfig is cluster logging component configuration.

  Enums:
    EnableComponentsValueListEntryValuesEnum:

  Fields:
    enableComponents: Select components to collect logs. An empty set would
      disable all logging.
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
      ADDON_MANAGER: kube-addon-manager
    """
        COMPONENT_UNSPECIFIED = 0
        SYSTEM_COMPONENTS = 1
        WORKLOADS = 2
        APISERVER = 3
        SCHEDULER = 4
        CONTROLLER_MANAGER = 5
        ADDON_MANAGER = 6
    enableComponents = _messages.EnumField('EnableComponentsValueListEntryValuesEnum', 1, repeated=True)