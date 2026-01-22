from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkingConfig(_messages.Message):
    """Configuration options for networking connections in the Composer 2
  environment.

  Enums:
    ConnectionTypeValueValuesEnum: Optional. Indicates the user requested
      specifc connection type between Tenant and Customer projects. You cannot
      set networking connection type in public IP environment.

  Fields:
    connectionType: Optional. Indicates the user requested specifc connection
      type between Tenant and Customer projects. You cannot set networking
      connection type in public IP environment.
  """

    class ConnectionTypeValueValuesEnum(_messages.Enum):
        """Optional. Indicates the user requested specifc connection type between
    Tenant and Customer projects. You cannot set networking connection type in
    public IP environment.

    Values:
      CONNECTION_TYPE_UNSPECIFIED: No specific connection type was requested,
        so the environment uses the default value corresponding to the rest of
        its configuration.
      VPC_PEERING: Requests the use of VPC peerings for connecting the
        Customer and Tenant projects.
      PRIVATE_SERVICE_CONNECT: Requests the use of Private Service Connect for
        connecting the Customer and Tenant projects.
    """
        CONNECTION_TYPE_UNSPECIFIED = 0
        VPC_PEERING = 1
        PRIVATE_SERVICE_CONNECT = 2
    connectionType = _messages.EnumField('ConnectionTypeValueValuesEnum', 1)