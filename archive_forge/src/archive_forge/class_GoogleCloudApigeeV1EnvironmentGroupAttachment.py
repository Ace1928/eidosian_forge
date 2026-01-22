from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1EnvironmentGroupAttachment(_messages.Message):
    """EnvironmentGroupAttachment is a resource which defines an attachment of
  an environment to an environment group.

  Enums:
    StateValueValuesEnum: Output only. State of the environment group
      attachment. Values other than ACTIVE means the resource is not ready to
      use.

  Fields:
    createdAt: Output only. The time at which the environment group attachment
      was created as milliseconds since epoch.
    environment: Required. ID of the attached environment.
    environmentGroupId: Output only. ID of the environment group.
    name: ID of the environment group attachment.
    state: Output only. State of the environment group attachment. Values
      other than ACTIVE means the resource is not ready to use.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the environment group attachment. Values other
    than ACTIVE means the resource is not ready to use.

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
    environment = _messages.StringField(2)
    environmentGroupId = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)