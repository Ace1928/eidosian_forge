from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ModuleEnablementStateValueValuesEnum(_messages.Enum):
    """The state of enablement for the module at its level of the resource
    hierarchy.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Default value. This value is unused.
      INHERITED: State is inherited from the parent resource.
      ENABLED: State is enabled.
      DISABLED: State is disabled.
    """
    ENABLEMENT_STATE_UNSPECIFIED = 0
    INHERITED = 1
    ENABLED = 2
    DISABLED = 3