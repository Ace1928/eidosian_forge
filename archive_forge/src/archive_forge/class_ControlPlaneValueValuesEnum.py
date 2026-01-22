from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ControlPlaneValueValuesEnum(_messages.Enum):
    """Deprecated: use `management` instead Enables automatic control plane
    management.

    Values:
      CONTROL_PLANE_MANAGEMENT_UNSPECIFIED: Unspecified
      AUTOMATIC: Google should provision a control plane revision and make it
        available in the cluster. Google will enroll this revision in a
        release channel and keep it up to date. The control plane revision may
        be a managed service, or a managed install.
      MANUAL: User will manually configure the control plane (e.g. via CLI, or
        via the ControlPlaneRevision KRM API)
    """
    CONTROL_PLANE_MANAGEMENT_UNSPECIFIED = 0
    AUTOMATIC = 1
    MANUAL = 2