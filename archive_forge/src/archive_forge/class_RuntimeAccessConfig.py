from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeAccessConfig(_messages.Message):
    """Specifies the login configuration for Runtime

  Enums:
    AccessTypeValueValuesEnum: The type of access mode this instance.

  Fields:
    accessType: The type of access mode this instance.
    proxyUri: Output only. The proxy endpoint that is used to access the
      runtime.
    runtimeOwner: The owner of this runtime after creation. Format:
      `alias@example.com` Currently supports one owner only.
  """

    class AccessTypeValueValuesEnum(_messages.Enum):
        """The type of access mode this instance.

    Values:
      RUNTIME_ACCESS_TYPE_UNSPECIFIED: Unspecified access.
      SINGLE_USER: Single user login.
      SERVICE_ACCOUNT: Service Account mode. In Service Account mode, Runtime
        creator will specify a SA that exists in the consumer project. Using
        Runtime Service Account field. Users accessing the Runtime need ActAs
        (Service Account User) permission.
    """
        RUNTIME_ACCESS_TYPE_UNSPECIFIED = 0
        SINGLE_USER = 1
        SERVICE_ACCOUNT = 2
    accessType = _messages.EnumField('AccessTypeValueValuesEnum', 1)
    proxyUri = _messages.StringField(2)
    runtimeOwner = _messages.StringField(3)