from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceMeshCondition(_messages.Message):
    """Condition being reported.

  Enums:
    CodeValueValuesEnum: Unique identifier of the condition which describes
      the condition recognizable to the user.
    SeverityValueValuesEnum: Severity level of the condition.

  Fields:
    code: Unique identifier of the condition which describes the condition
      recognizable to the user.
    details: A short summary about the issue.
    documentationLink: Links contains actionable information.
    severity: Severity level of the condition.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Unique identifier of the condition which describes the condition
    recognizable to the user.

    Values:
      CODE_UNSPECIFIED: Default Unspecified code
      MESH_IAM_PERMISSION_DENIED: Mesh IAM permission denied error code
      CNI_CONFIG_UNSUPPORTED: CNI config unsupported error code
      GKE_SANDBOX_UNSUPPORTED: GKE sandbox unsupported error code
      NODEPOOL_WORKLOAD_IDENTITY_FEDERATION_REQUIRED: Nodepool workload
        identity federation required error code
      CNI_INSTALLATION_FAILED: CNI installation failed error code
      CNI_POD_UNSCHEDULABLE: CNI pod unschedulable error code
      UNSUPPORTED_MULTIPLE_CONTROL_PLANES: Multiple control planes unsupported
        error code
    """
        CODE_UNSPECIFIED = 0
        MESH_IAM_PERMISSION_DENIED = 1
        CNI_CONFIG_UNSUPPORTED = 2
        GKE_SANDBOX_UNSUPPORTED = 3
        NODEPOOL_WORKLOAD_IDENTITY_FEDERATION_REQUIRED = 4
        CNI_INSTALLATION_FAILED = 5
        CNI_POD_UNSCHEDULABLE = 6
        UNSUPPORTED_MULTIPLE_CONTROL_PLANES = 7

    class SeverityValueValuesEnum(_messages.Enum):
        """Severity level of the condition.

    Values:
      SEVERITY_UNSPECIFIED: Unspecified severity
      ERROR: Indicates an issue that prevents the mesh from operating
        correctly
      WARNING: Indicates a setting is likely wrong, but the mesh is still able
        to operate
      INFO: An informational message, not requiring any action
    """
        SEVERITY_UNSPECIFIED = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    details = _messages.StringField(2)
    documentationLink = _messages.StringField(3)
    severity = _messages.EnumField('SeverityValueValuesEnum', 4)