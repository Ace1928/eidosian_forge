from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EnableComponentsValueListEntryValuesEnum(_messages.Enum):
    """EnableComponentsValueListEntryValuesEnum enum type.

    Values:
      COMPONENT_UNSPECIFIED: No component is specified
      SYSTEM_COMPONENTS: This indicates that system logging components is
        enabled.
      WORKLOADS: This indicates that user workload logging component is
        enabled.
    """
    COMPONENT_UNSPECIFIED = 0
    SYSTEM_COMPONENTS = 1
    WORKLOADS = 2