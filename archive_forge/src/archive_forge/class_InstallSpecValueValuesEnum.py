from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstallSpecValueValuesEnum(_messages.Enum):
    """The install_spec represents the intended state specified by the latest
    request that mutated install_spec in the feature spec, not the lifecycle
    state of the feature observed by the Hub feature controller that is
    reported in the feature state.

    Values:
      INSTALL_SPEC_UNSPECIFIED: Spec is unknown.
      INSTALL_SPEC_NOT_INSTALLED: Request to uninstall Policy Controller.
      INSTALL_SPEC_ENABLED: Request to install and enable Policy Controller.
      INSTALL_SPEC_SUSPENDED: Request to suspend Policy Controller i.e. its
        webhooks. If Policy Controller is not installed, it will be installed
        but suspended.
      INSTALL_SPEC_DETACHED: Request to stop all reconciliation actions by
        PoCo Hub controller. This is a breakglass mechanism to stop PoCo Hub
        from affecting cluster resources.
    """
    INSTALL_SPEC_UNSPECIFIED = 0
    INSTALL_SPEC_NOT_INSTALLED = 1
    INSTALL_SPEC_ENABLED = 2
    INSTALL_SPEC_SUSPENDED = 3
    INSTALL_SPEC_DETACHED = 4