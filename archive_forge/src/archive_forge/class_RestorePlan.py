from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestorePlan(_messages.Message):
    """The configuration of a potential series of Restore operations to be
  performed against Backups belong to a particular BackupPlan.

  Enums:
    StateValueValuesEnum: Output only. State of the RestorePlan. This State
      field reflects the various stages a RestorePlan can be in during the
      Create operation.

  Messages:
    LabelsValue: Optional. A set of custom labels supplied by user.

  Fields:
    backupPlan: Required. Immutable. A reference to the BackupPlan from which
      Backups may be used as the source for Restores created via this
      RestorePlan. Format: `projects/*/locations/*/backupPlans/*`.
    cluster: Required. Immutable. The target cluster into which Restores
      created via this RestorePlan will restore data. NOTE: the cluster's
      region must be the same as the RestorePlan. Valid formats: -
      `projects/*/locations/*/clusters/*` - `projects/*/zones/*/clusters/*`
    createTime: Output only. The timestamp when this RestorePlan resource was
      created.
    description: Optional. User specified descriptive string for this
      RestorePlan.
    etag: Output only. `etag` is used for optimistic concurrency control as a
      way to help prevent simultaneous updates of a restore from overwriting
      each other. It is strongly suggested that systems make use of the `etag`
      in the read-modify-write cycle to perform restore updates in order to
      avoid race conditions: An `etag` is returned in the response to
      `GetRestorePlan`, and systems are expected to put that etag in the
      request to `UpdateRestorePlan` or `DeleteRestorePlan` to ensure that
      their change will be applied to the same version of the resource.
    labels: Optional. A set of custom labels supplied by user.
    name: Output only. The full name of the RestorePlan resource. Format:
      `projects/*/locations/*/restorePlans/*`.
    restoreConfig: Required. Configuration of Restores created via this
      RestorePlan.
    state: Output only. State of the RestorePlan. This State field reflects
      the various stages a RestorePlan can be in during the Create operation.
    stateReason: Output only. Human-readable description of why RestorePlan is
      in the current `state`
    uid: Output only. Server generated global unique identifier of
      [UUID](https://en.wikipedia.org/wiki/Universally_unique_identifier)
      format.
    updateTime: Output only. The timestamp when this RestorePlan resource was
      last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the RestorePlan. This State field reflects the
    various stages a RestorePlan can be in during the Create operation.

    Values:
      STATE_UNSPECIFIED: Default first value for Enums.
      CLUSTER_PENDING: Waiting for cluster state to be RUNNING.
      READY: The RestorePlan has successfully been created and is ready for
        Restores.
      FAILED: RestorePlan creation has failed.
      DELETING: The RestorePlan is in the process of being deleted.
    """
        STATE_UNSPECIFIED = 0
        CLUSTER_PENDING = 1
        READY = 2
        FAILED = 3
        DELETING = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. A set of custom labels supplied by user.

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
    backupPlan = _messages.StringField(1)
    cluster = _messages.StringField(2)
    createTime = _messages.StringField(3)
    description = _messages.StringField(4)
    etag = _messages.StringField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    name = _messages.StringField(7)
    restoreConfig = _messages.MessageField('RestoreConfig', 8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    stateReason = _messages.StringField(10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)