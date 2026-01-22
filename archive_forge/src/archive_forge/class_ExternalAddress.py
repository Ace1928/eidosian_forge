from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalAddress(_messages.Message):
    """Represents an allocated external IP address and its corresponding
  internal IP address in a private cloud.

  Enums:
    StateValueValuesEnum: Output only. The state of the resource.

  Fields:
    createTime: Output only. Creation time of this resource.
    description: User-provided description for this resource.
    externalIp: Output only. The external IP address of a workload VM.
    internalIp: The internal IP address of a workload VM.
    name: Output only. The resource name of this external IP address. Resource
      names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/externalAddresses/my-address`
    state: Output only. The state of the resource.
    uid: Output only. System-generated unique identifier for the resource.
    updateTime: Output only. Last update time of this resource.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the resource.

    Values:
      STATE_UNSPECIFIED: The default value. This value should never be used.
      ACTIVE: The address is ready.
      CREATING: The address is being created.
      UPDATING: The address is being updated.
      DELETING: The address is being deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        UPDATING = 3
        DELETING = 4
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    externalIp = _messages.StringField(3)
    internalIp = _messages.StringField(4)
    name = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    uid = _messages.StringField(7)
    updateTime = _messages.StringField(8)