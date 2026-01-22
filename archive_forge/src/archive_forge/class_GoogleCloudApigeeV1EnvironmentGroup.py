from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1EnvironmentGroup(_messages.Message):
    """EnvironmentGroup configuration. An environment group is used to group
  one or more Apigee environments under a single host name.

  Enums:
    StateValueValuesEnum: Output only. State of the environment group. Values
      other than ACTIVE means the resource is not ready to use.

  Fields:
    createdAt: Output only. The time at which the environment group was
      created as milliseconds since epoch.
    hostnames: Required. Host names for this environment group.
    lastModifiedAt: Output only. The time at which the environment group was
      last updated as milliseconds since epoch.
    name: ID of the environment group.
    state: Output only. State of the environment group. Values other than
      ACTIVE means the resource is not ready to use.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the environment group. Values other than ACTIVE
    means the resource is not ready to use.

    Values:
      STATE_UNSPECIFIED: Resource is in an unspecified state.
      CREATING: Resource is being created.
      ACTIVE: Resource is provisioned and ready to use.
      DELETING: The resource is being deleted.
      UPDATING: The resource is being updated.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        UPDATING = 4
    createdAt = _messages.IntegerField(1)
    hostnames = _messages.StringField(2, repeated=True)
    lastModifiedAt = _messages.IntegerField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)