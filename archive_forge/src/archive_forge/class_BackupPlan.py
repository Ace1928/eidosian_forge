from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupPlan(_messages.Message):
    """A `BackupPlan` specifies some common fields, such as `display_name` as
  well as one or more `BackupRule` messages. Each `BackupRule` has a retention
  policy and defines a schedule by which the system is to perform backup
  workloads.

  Enums:
    StateValueValuesEnum: Output only. The `State` for the `BackupPlan`.

  Messages:
    LabelsValue: Optional. This collection of key/value pairs allows for
      custom labels to be supplied by the user. Example, {"tag": "Weekly"}.

  Fields:
    adminScope: Optional. TODO b/325560313: Deprecated and field will be
      removed after UI integration change. The resource name of AdminScope
      with which this `BackupPlan` is associated. Format :
      projects/{project}/locations/{location}/adminScopes/{admin_scope}
    backupRules: Required. The backup rules for this `BackupPlan`. There must
      be at least one `BackupRule` message.
    createTime: Output only. When the `BackupPlan` was created.
    description: Optional. The description of the `BackupPlan` resource. The
      description allows for additional details about `BackupPlan` and its use
      cases to be provided. An example description is the following: "This is
      a backup plan that performs a daily backup at 6pm and retains data for 3
      months". The description must be at most 2048 characters.
    displayName: Optional. TODO b/325560313: Deprecated and field will be
      removed after UI integration change. The display name of the
      `BackupPlan`.
    etag: Optional. `etag` is returned from the service in the response. As a
      user of the service, you may provide an etag value in this field to
      prevent stale resources.
    labels: Optional. This collection of key/value pairs allows for custom
      labels to be supplied by the user. Example, {"tag": "Weekly"}.
    name: Output only. The resource name of the `BackupPlan`. Format:
      `projects/{project}/locations/{location}/backupPlans/{backup_plan}`
    resourceType: Required. The resource type to which the `BackupPlan` will
      be applied. Examples include, "compute.googleapis.com/Instance" and
      "storage.googleapis.com/Bucket".
    state: Output only. The `State` for the `BackupPlan`.
    updateTime: Output only. When the `BackupPlan` was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The `State` for the `BackupPlan`.

    Values:
      STATE_UNSPECIFIED: State not set.
      CREATING: The resource is being created.
      ACTIVE: The resource has been created and is fully usable.
      DELETING: The resource is being deleted.
      READY: TODO b/325560313: Deprecated and field will be removed after UI
        integration change. The resource has been created.
      INACTIVE: The resource has been created but is not usable.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        READY = 4
        INACTIVE = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. This collection of key/value pairs allows for custom labels
    to be supplied by the user. Example, {"tag": "Weekly"}.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    adminScope = _messages.StringField(1)
    backupRules = _messages.MessageField('BackupRule', 2, repeated=True)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    displayName = _messages.StringField(5)
    etag = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    name = _messages.StringField(8)
    resourceType = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    updateTime = _messages.StringField(11)